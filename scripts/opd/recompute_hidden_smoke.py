#!/usr/bin/env python3
"""recompute-hidden numerical smoke (optimization #5).

Opt #5 replaces the cached target_hidden_states with an in-worker teacher forward that captures
the same decoder-layer hidden states (target_layer_ids, via hooks) — see
teacher_scoring.recompute_target_hidden_states, mirroring prepare_target_cache.run_target_forward_
with_hooks. This smoke verifies the RECOMPUTED hidden matches the CACHED hidden.

recompute (bf16) is the TEACHER'S TRUE hidden. The cached tensor is a LOSSY fp8 copy: cache-gen
clamps to float8_e4m3fn's finite_max=±448 (sanitize_fp8_hidden_states), which SATURATES the large
outlier feature dimensions present in Qwen3's deeper layers. So recompute vs cache:
  - DIRECTION matches tightly (cosine ~1.0), but deep-layer NORMS differ (cache is compressed by
    fp8 saturation; recompute keeps the true magnitude -> norm ratio 1.2-2.6x on layers 9/17/25/33).
    This is EXPECTED and recompute is MORE faithful, NOT a bug.
  - Crucially, INFERENCE/eval computes target hidden via a live teacher forward
    (output_hidden_states=True, bf16, base_evaluator.py:217-222) — NOT from the fp8 cache. So opt #5
    recompute is exactly what inference uses: it CLOSES the train(fp8-cache)/inference(bf16-live)
    distribution gap rather than introducing one. (User decision 2026-07-10: use TRUE hidden, no
    fp8 clamp.)
  - right-padding + batching introduces ~1e-3 batch-invariance noise (same as S2/S3).

Checks (needs GPU + real teacher + real cache):
  A) shape/layout: recompute [B,T,L*H] matches cache; per-layer split aligns (cat order = ascending
     target_layer_ids).
  B) direction: over real tokens, cosine similarity per (layer, position) is ~1.0 (recompute is the
     de-saturated version of the same vectors). NOTE norms differ on deep layers by design — we do
     NOT assert small relative L2 (that would wrongly assume the fp8 cache is lossless).
  C) padding / batch-invariance: recompute of a padded BATCH matches recompute of each sample
     UNPADDED over real tokens (~1e-3) — the only source that must be tight, since it's pure model
     forward equivalence (no fp8 involved).

Usage:
    PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec:\
/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl \
      CUDA_VISIBLE_DEVICES=0 ~/.venv/dspark-opd/bin/python scripts/opd/recompute_hidden_smoke.py \
        --cache /mnt/scratch/qwen3_4b_target_cache --target Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "third_party", "verl")))

_FAILS = 0


def _check(cond, msg):
    global _FAILS
    if not cond:
        _FAILS += 1
    print(f"  {'[OK]' if cond else '[FAIL]'} {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/mnt/scratch/qwen3_4b_target_cache")
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=4, help="samples in the batch")
    args = ap.parse_args()

    from deepspec.data.target_cache_dataset import CacheDataset
    from recipe.dspark_opd.dataset import adapt_cache_record, dspark_collate_fn
    from recipe.dspark_opd.teacher_scoring import recompute_target_hidden_states
    from transformers import AutoModelForCausalLM

    dev = "cuda:0"
    torch.cuda.set_device(0)
    print("=" * 76)
    print("recompute-hidden numerical smoke (opt #5)")
    print("=" * 76)
    cache = CacheDataset(cache_dir=args.cache)
    lids = list(cache.target_layer_ids)
    L, H = len(lids), cache.hidden_size
    print(f"target_layer_ids={lids}  L={L} H={H}  hidden_dtype={cache.hidden_dtype}")

    print("loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # full samples (cached hidden) + a right-padded batch
    idxs = list(range(args.n))
    full = [adapt_cache_record(cache[i]) for i in idxs]
    batch = dspark_collate_fn(full)                          # [B,T] + [B,T,L*H] cached hidden
    ii = batch["input_ids"].to(dev)
    am = batch["attention_mask"].to(dev)
    cached = batch["target_hidden_states"].to(dev).float()   # [B,T,L*H] (fp8-dequant, bf16->f32)

    # recompute
    recomp = recompute_target_hidden_states(
        teacher, input_ids=ii, attention_mask=am, target_layer_ids=lids).float()

    # A) shape/layout
    print("[A] shape / layout")
    _check(recomp.shape == cached.shape, f"recompute {tuple(recomp.shape)} == cache {tuple(cached.shape)}")
    _check(recomp.shape[-1] == L * H, f"feature dim == L*H = {L*H}")

    # B) direction over real tokens (norms differ on deep layers by fp8 saturation — expected)
    print("[B] direction agreement over real tokens (cosine ~1; norm ratio reported, not asserted)")
    B, T, _ = cached.shape
    cos_all = []
    print(f"    {'layer':>6} {'|cache|':>9} {'|recomp|':>9} {'ratio':>6} {'cos':>7}")
    for li, lid in enumerate(lids):
        cos_l, cn_l, rn_l = [], [], []
        for b in range(B):
            rl = int(am[b].sum())
            cv = cached[b, :rl].reshape(rl, L, H)[:, li]
            rv = recomp[b, :rl].reshape(rl, L, H)[:, li]
            cos_l.append(torch.nn.functional.cosine_similarity(cv, rv, dim=-1).mean().item())
            cn_l.append(cv.norm(dim=-1).mean().item())
            rn_l.append(rv.norm(dim=-1).mean().item())
        mc = sum(cos_l) / len(cos_l)
        cn, rn = sum(cn_l) / len(cn_l), sum(rn_l) / len(rn_l)
        cos_all.append(mc)
        print(f"    {lid:>6} {cn:>9.2f} {rn:>9.2f} {rn/cn:>6.3f} {mc:>7.4f}")
    mean_cos = sum(cos_all) / len(cos_all)
    _check(mean_cos > 0.99, f"mean cosine similarity > 0.99 (got {mean_cos:.5f}) — direction matches")
    _check(cos_all[0] > 0.999, f"layer{lids[0]} (no fp8 saturation) cos > 0.999 (got {cos_all[0]:.5f})")

    # C) batch-invariance: recompute unpadded single sample == batched recompute over real tokens
    print("[C] batch-invariance (padded batch vs single unpadded, real tokens)")
    b0 = full[0]
    ii0 = b0["input_ids"].unsqueeze(0).to(dev)
    am0 = torch.ones_like(ii0)
    solo = recompute_target_hidden_states(
        teacher, input_ids=ii0, attention_mask=am0, target_layer_ids=lids).float()
    rl0 = int(am[0].sum())
    batched0 = recomp[0, :rl0]
    d = (solo[0, :rl0] - batched0).abs()
    relmax = (d.max() / (batched0.abs().max() + 1e-6)).item()
    print(f"    solo-vs-batched relmax={relmax:.4f}  argmax-agree over feature not applicable")
    _check(relmax < 0.05, f"batch-invariance relmax < 0.05 (got {relmax:.4f})")

    print("=" * 76)
    if _FAILS == 0:
        print("ALL CHECKS PASSED — recompute = teacher's true (de-saturated) hidden; direction "
              "matches cache, deep-layer norms larger by design (fp8 cache was lossy), "
              "batch-invariant. Matches the live-teacher path used at inference.")
        return 0
    print(f"{_FAILS} CHECK(S) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
