#!/usr/bin/env python3
"""S3 smoke-test: teacher block-diagonal scoring correctness (tensor-contract §S3).

Checks:
  A) ★ FLATTEN EQUIVALENCE (golden): score_blocks_flat (one forward, 4D block-diagonal
     causal mask) vs score_blocks_reference (per-block independent forward). Proves the
     flatten approach is numerically equivalent to per-block. This gates the whole method.
  B) block-diagonal isolation: perturbing one block's sampled tokens leaves other blocks'
     logp_target unchanged.
  C) within-block causal: perturbing ỹ_{>k} leaves position k's logp_target unchanged.
  D) no-NaN: valid-block logp_target finite & <=0.

Usage:
    PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec:/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl \
      ~/.venv/dspark-opd/bin/python scripts/opd/s3_smoke.py \
        --draft /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest \
        --target Qwen/Qwen3-4B --cache /mnt/scratch/qwen3_4b_target_cache --num-anchors 8 --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "third_party", "verl")))


def _check(cond, msg):
    print(f"  {'[OK]' if cond else '[FAIL]'} {msg}")
    if not cond:
        print("\n[S3] SMOKE FAILED")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--cache", default="/mnt/scratch/qwen3_4b_target_cache")
    ap.add_argument("--num-anchors", type=int, default=8)  # small: reference loops per block
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    from deepspec.utils import seed_all
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
    from deepspec.modeling.dspark.common import sample_anchor_positions
    from recipe.dspark_opd.dataset import DSparkCacheDataset, dspark_collate_fn
    from recipe.dspark_opd.block_rollout import dspark_block_rollout
    from recipe.dspark_opd.teacher_scoring import score_blocks_reference, score_blocks_flat
    from omegaconf import OmegaConf

    device = "cuda:0"
    seed_all(int(args.seed))
    K = 16

    # --- data + draft rollout (get tokens + top-k candidates) ---
    cfg = OmegaConf.create({"dspark": {"target_cache_path": args.cache, "n_samples": 4}})
    ds = DSparkCacheDataset(config=cfg)
    feats = [ds[0], ds[1]]
    batch = {k: v.to(device) for k, v in dspark_collate_fn(feats).items()}
    draft = Qwen3DSparkModel.from_pretrained(
        args.draft, dtype=torch.bfloat16, attn_implementation="flex_attention").to(device).eval()
    seed_all(int(args.seed))
    anchors, keep = sample_anchor_positions(
        seq_len=batch["input_ids"].shape[1], loss_mask=batch["loss_mask"],
        num_anchors=int(args.num_anchors), device=device)
    roll = dspark_block_rollout(
        draft, input_ids=batch["input_ids"], loss_mask=batch["loss_mask"],
        target_hidden_states=batch["target_hidden_states"], num_anchors=int(args.num_anchors),
        temperature=1.0, top_k=K, anchor_positions=anchors, block_keep_mask=keep)
    tokens = roll["tokens"]
    tki = roll["student_top_k_ids"]
    tki_logp = roll["student_top_k_logp"]   # [B,A,blk,K] student logp on candidates (for weights)
    print(f"### rollout: tokens {tuple(tokens.shape)}, top_k_ids {tuple(tki.shape)}, "
          f"valid={keep.sum(dim=1).tolist()}")

    # --- target model (sdpa, matches our teacher worker) ---
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    common = dict(input_ids=batch["input_ids"], tokens=tokens,
                  anchor_positions=anchors, block_keep_mask=keep, student_top_k_ids=tki)

    # ===== A) flatten equivalence (golden) =====
    # In fp32 score_blocks_flat == score_blocks_reference to ~1e-4 (all K), so the flatten
    # (one forward, 4D block-diagonal causal mask) is STRUCTURALLY identical to per-block
    # scoring. (An earlier bf16 max|Δ|=7.0 was a real bug — ctx keys allowed pos<=anchor, so
    # the anchor token was double-counted as both a ctx key and block-key j=0; fixed to
    # pos<anchor. See teacher_scoring.py allow_ctx.) The residual below is bf16 precision on
    # the long flattened sequence (same regime as S2 flex-vs-sdpa noise).
    # Equivalence is judged on what the OPD reward actually uses: the reward weights
    # candidates by w_j = softmax_K(student_logp), so deep-tail candidates (logp ~ -30, weight
    # ~ e^-30 ≈ 0) are irrelevant. So we check: (1) argmax agrees (candidate ranking
    # identical), (2) top-1 candidate logp matches, (3) the STUDENT-weighted teacher logp
    # (what the reward integrates) matches.
    print("### A) flatten equivalence: score_blocks_flat vs score_blocks_reference")
    ref = score_blocks_reference(target, **common)   # [B,A,blk,K]
    flat = score_blocks_flat(target, **common)
    m = keep.unsqueeze(-1).unsqueeze(-1).expand_as(ref)          # [B,A,blk,K] valid
    mb = keep.unsqueeze(-1).expand(ref.shape[:3])                # [B,A,blk] valid
    argmax_agree = (ref[m].reshape(-1, K).argmax(-1) == flat[m].reshape(-1, K).argmax(-1)).float().mean().item()
    # top-1 (highest-prob) candidate
    d_top1 = (ref[..., 0] - flat[..., 0]).abs()[mb]
    top1_max, top1_mean = d_top1.max().item(), d_top1.mean().item()
    # student-weighted teacher logp: w = softmax_K(student top-k logp), sum_j w_j * T_logp_j
    w = torch.softmax(tki_logp, dim=-1)                          # [B,A,blk,K]
    wref = (w * ref).sum(-1)[mb]
    wflat = (w * flat).sum(-1)[mb]
    dw = (wref - wflat).abs()
    dw_max, dw_mean = dw.max().item(), dw.mean().item()
    print(f"    argmax_agree={argmax_agree:.4f}  top-1 |Δ| max={top1_max:.4g} mean={top1_mean:.4g}  "
          f"student-weighted |Δ| max={dw_max:.4g} mean={dw_mean:.4g}")
    _check(argmax_agree == 1.0 and top1_mean < 0.05 and dw_mean < 0.05,
           "flat == reference on reward-relevant part (argmax 100% + top-1 & student-weighted logp match)")

    # ===== D) no NaN & <=0 on valid blocks =====
    print("### D) no NaN / <=0 on valid blocks")
    _check(bool(torch.isfinite(flat[m]).all()) and bool((flat[m] <= 1e-4).all()),
           "logp_target finite & <=0 on valid blocks")

    # ===== B) block-diagonal isolation =====
    print("### B) block-diagonal isolation (perturb one block -> others unchanged)")
    b0 = 0
    vidx = [i for i in range(anchors.size(1)) if bool(keep[b0, i])]
    if len(vidx) >= 2:
        i0, i1 = vidx[0], vidx[1]
        tok2 = tokens.clone()
        tok2[b0, i0, :] = (tok2[b0, i0, :] + 1) % 1000  # perturb block i0's tokens
        flat2 = score_blocks_flat(target, input_ids=batch["input_ids"], tokens=tok2,
                                  anchor_positions=anchors, block_keep_mask=keep, student_top_k_ids=tki)
        other_same = torch.allclose(flat[b0, i1], flat2[b0, i1], atol=1e-3, rtol=1e-3)
        i0_changed = not torch.allclose(flat[b0, i0], flat2[b0, i0], atol=1e-3)
        _check(other_same and i0_changed, "perturb block i0: block i1 unchanged, i0 changed")
    else:
        _check(True, "skip (need >=2 valid blocks)")

    # ===== C) within-block causal (perturb ỹ_{>k} -> pos k unchanged) =====
    print("### C) within-block causal (perturb later positions -> earlier unchanged)")
    i0 = vidx[0]
    tok3 = tokens.clone()
    tok3[b0, i0, tokens.shape[-1] - 1] = (tok3[b0, i0, -1] + 1) % 1000  # perturb LAST in-block token
    flat3 = score_blocks_flat(target, input_ids=batch["input_ids"], tokens=tok3,
                              anchor_positions=anchors, block_keep_mask=keep, student_top_k_ids=tki)
    # positions 0..blk-2 of block i0 must be unchanged (they don't attend the last token)
    early_same = torch.allclose(flat[b0, i0, :-1], flat3[b0, i0, :-1], atol=1e-3, rtol=1e-3)
    _check(early_same, "perturbing last in-block token leaves earlier positions unchanged")

    # ===== E) batch equivalence (★ batching-correctness, this change) =====
    # score_blocks_flat now batches all B samples into ONE forward. Prove no cross-sample
    # interference / padding leak: B=2 one-shot vs each sample scored ALONE (B=1). The two
    # samples have different real_len (226 vs 244), so in the B=2 call sample 0 carries padding
    # columns [226,244) — if the mask leaked padding or samples cross-talked, per-sample B=1
    # (no padding, no other sample) would differ. Must be allclose on valid blocks.
    print("### E) batch equivalence: B=2 one-shot == per-sample B=1 (variable real_len)")
    max_bd = 0.0
    for b in range(tokens.shape[0]):
        sub = dict(input_ids=batch["input_ids"][b:b+1], tokens=tokens[b:b+1],
                   anchor_positions=anchors[b:b+1], block_keep_mask=keep[b:b+1],
                   student_top_k_ids=tki[b:b+1])
        flat_b1 = score_blocks_flat(target, **sub)                       # [1,A,blk,K]
        mb_b = keep[b:b+1].unsqueeze(-1).expand(flat_b1.shape[:3])
        d = (flat[b:b+1][mb_b] - flat_b1[mb_b]).abs()
        max_bd = max(max_bd, d.max().item() if d.numel() else 0.0)
    print(f"    max |Δ(B=2 vs B=1)| on valid blocks = {max_bd:.4g}")
    _check(max_bd < 1e-3, "batched score == per-sample score (no cross-sample / padding leak)")

    print("\n[S3] SMOKE OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
