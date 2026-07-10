#!/usr/bin/env python3
"""Fused-step smoke: the FUSED train_step sequence (rollout→teacher→update, with per-micro
slicing) == a whole-batch reference on the SAME rollout+teacher outputs.

The fusion (worker.train_step) reuses the SAME validated pure functions as the separate path
(dspark_block_rollout / score_blocks_flat / loss_bridge), so numerics are equal BY CONSTRUCTION.
The genuine new risk is WIRING: train_step re-implements the update micro-loop with its own
per-micro slicing (input_ids[sl], tokens[sl], T_on_S[sl], ...). A slicing/indexing bug there
would silently mis-pair a sample's rollout/teacher tensors. This smoke catches that by:

  A) run rollout ONCE (seeded) + teacher ONCE → fixed (tokens, top-k, T_on_S) for the batch.
  B) REFERENCE: whole-batch update (micro = full B) → grads g_ref.
  C) FUSED-STYLE: the exact per-micro slicing train_step uses (micro=1, 1/n_micro scale,
     accumulate) → grads g_fused.
  D) assert g_ref ≈ g_fused (bit-exact-ish: fusion only reorders into micros; == s4 test F but
     starting from a live rollout, exercising the [sl] slicing of ALL rollout/teacher tensors).
  E) determinism: same seed twice → identical loss.
  F) trainable grads finite & nonzero; frozen embed/lm_head no grad.

Usage:
    PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec:/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl \
      ~/.venv/dspark-opd/bin/python scripts/opd/fused_step_smoke.py \
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
        print("\n[FUSED] SMOKE FAILED")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--cache", default="/mnt/scratch/qwen3_4b_target_cache")
    ap.add_argument("--num-anchors", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    from omegaconf import OmegaConf
    from deepspec.utils import seed_all
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
    from recipe.dspark_opd.dataset import DSparkCacheDataset, dspark_collate_fn
    from recipe.dspark_opd.block_rollout import dspark_block_rollout
    from recipe.dspark_opd.teacher_scoring import score_blocks_flat
    from recipe.dspark_opd.loss_bridge import (
        logp_on_topk_ids, build_opd_reward, confidence_accept_rate_topk,
        confidence_bce, block_decay_weight_mask, flatten_blocks_to_sequence)
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn, compute_policy_loss_vanilla

    device = "cuda:0"
    seed_all(int(args.seed))
    K, A, GAMMA, temp = 16, int(args.num_anchors), 4.0, 1.0
    cfg_pl = OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": 0.2,
                               "clip_ratio_high": 0.2, "clip_ratio_c": 3.0})
    adv_fn = get_adv_estimator_fn("token_reward_direct")

    cfg = OmegaConf.create({"dspark": {"target_cache_path": args.cache, "n_samples": 4}})
    ds = DSparkCacheDataset(config=cfg)
    batch = {k: v.to(device) for k, v in dspark_collate_fn([ds[0], ds[1]]).items()}
    draft = Qwen3DSparkModel.from_pretrained(
        args.draft, dtype=torch.bfloat16, attn_implementation="flex_attention").to(device)
    draft.set_embedding_head_trainable(False)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    # ---- one micro's loss (mirrors train_step's per-slice body exactly) ----
    def _micro_loss(sl, tokens, anchors, keep, tki, S_old, T_on_S, eval_mask):
        blk = tokens.shape[2]
        ii = batch["input_ids"][sl]
        anc = anchors[sl]
        anchor_tok = torch.gather(ii, 1, anc)
        prev = torch.cat([anchor_tok.unsqueeze(-1), tokens[sl][:, :, : blk - 1]], dim=-1)
        out = draft(input_ids=ii,
                    target_hidden_states=batch["target_hidden_states"][sl].to(torch.bfloat16),
                    loss_mask=batch["loss_mask"][sl],
                    anchor_positions=anc, block_keep_mask=keep[sl], block_prev_tokens=prev)
        S_grad = logp_on_topk_ids(out.draft_logits, tki[sl], temperature=temp)
        rm = build_opd_reward(T_on_S[sl], S_old[sl], weight_mode="student_p")
        dmask = block_decay_weight_mask(eval_mask[sl], blk, GAMMA)
        mask_flat = flatten_blocks_to_sequence(dmask.unsqueeze(-1)).squeeze(-1)
        adv, _ = adv_fn(token_level_rewards=flatten_blocks_to_sequence(rm), response_mask=mask_flat)
        lp = flatten_blocks_to_sequence(S_grad)
        pg, _ = compute_policy_loss_vanilla(old_log_prob=lp.detach(), log_prob=lp, advantages=adv,
                                            response_mask=mask_flat, loss_agg_mode="token-mean", config=cfg_pl)
        acc = confidence_accept_rate_topk(S_old[sl], T_on_S[sl])
        cl = confidence_bce(out.confidence_pred, acc, dmask)
        return pg + 1.0 * cl

    def _rollout_and_teacher(seed):
        seed_all(seed)
        roll = dspark_block_rollout(draft, input_ids=batch["input_ids"], loss_mask=batch["loss_mask"],
                                    target_hidden_states=batch["target_hidden_states"].to(torch.bfloat16),
                                    num_anchors=A, temperature=temp, top_k=K)
        anchors, keep = roll["anchor_positions"], roll["block_keep_mask"].bool()
        tokens, tki, S_old, em = roll["tokens"], roll["student_top_k_ids"], roll["student_top_k_logp"], roll["eval_mask"].bool()
        T_on_S = score_blocks_flat(target, input_ids=batch["input_ids"], tokens=tokens,
                                   anchor_positions=anchors, block_keep_mask=keep, student_top_k_ids=tki)
        return tokens, anchors, keep, tki, S_old, T_on_S, em

    B = batch["input_ids"].shape[0]

    # ===== A) fixed rollout+teacher =====
    print("### A) rollout + teacher (fused sequence, seeded)")
    r = _rollout_and_teacher(int(args.seed))
    _check(torch.isfinite(r[5][r[2].unsqueeze(-1).expand(r[5].shape[:3])]).all(),
           f"rollout+teacher ran; T_on_S finite (valid={r[2].sum(1).tolist()})")

    # ===== B) whole-batch reference grads =====
    print("### B) whole-batch reference (micro = full B)")
    draft.zero_grad(set_to_none=True)
    _micro_loss(slice(0, B), *r).backward()
    g_ref = {n: p.grad.detach().clone() for n, p in draft.named_parameters() if p.grad is not None}

    # ===== C) fused-style per-micro slicing (micro=1, 1/n_micro accumulate) =====
    print("### C) fused-style per-micro (micro=1, 1/B scale, accumulate) == train_step body")
    draft.zero_grad(set_to_none=True)
    for i in range(B):
        (_micro_loss(slice(i, i + 1), *r) * (1.0 / B)).backward()
    g_fused = {n: p.grad.detach().clone() for n, p in draft.named_parameters() if p.grad is not None}

    # ===== D) grads match (wiring/slicing correct) =====
    max_gd = max((g_ref[n] - g_fused[n]).abs().max().item() for n in g_ref) if g_ref else 1e9
    _check(set(g_ref) == set(g_fused) and max_gd < 1e-2,
           f"per-micro sliced grads == whole-batch (max|Δ|={max_gd:.4g}; slicing of tokens/T_on_S/etc correct)")

    # ===== E) determinism =====
    print("### E) determinism (same seed -> same loss)")
    r2 = _rollout_and_teacher(int(args.seed))
    l1 = float(_micro_loss(slice(0, B), *r).detach())
    l2 = float(_micro_loss(slice(0, B), *r2).detach())
    _check(abs(l1 - l2) < 1e-4, f"same seed -> same loss ({l1:.5f} vs {l2:.5f})")

    # ===== F) trainable grads finite/nonzero; frozen no grad =====
    print("### F) trainable grads finite/nonzero; frozen embed/lm_head no grad")
    def _gn(prefixes):
        g = 0.0
        for n, p in draft.named_parameters():
            if any(n.startswith(pf) or f".{pf}" in n for pf in prefixes) and p.grad is not None:
                g += p.grad.float().pow(2).sum().item()
        return g ** 0.5
    train_gn = _gn(["layers", "fc", "markov_head", "confidence_head", "hidden_norm", "norm"])
    frozen_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                      for n, p in draft.named_parameters()
                      if n.startswith("embed_tokens") or n.startswith("lm_head"))
    all_finite = all(torch.isfinite(p.grad).all() for _, p in draft.named_parameters() if p.grad is not None)
    _check(all_finite and train_gn > 0 and not frozen_grad,
           f"trainable grad_norm={train_gn:.4g}>0 finite; frozen no grad")

    print("\n[FUSED] SMOKE OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
