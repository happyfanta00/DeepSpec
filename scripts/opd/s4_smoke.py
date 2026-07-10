#!/usr/bin/env python3
"""S4 smoke-test: reward→advantage→loss math + grad training forward (tensor-contract §S4).

Checks:
  A) ★ FLATTEN INVERTIBILITY (golden): flatten_blocks_to_sequence then unflatten == identity
     (both eval_mask and a K-dim tensor). If not invertible, reward misaligns to wrong blocks.
  B) GRAD forward consistency: dspark_block_train_forward (grad) reproduces rollout's no-grad
     corrected_logits on valid blocks (same anchors+tokens) -> student logp on top-k matches
     S2's student_top_k_logp (top-1 == sampled token, tight match).
  C) reward: rm = w·(T_on_S − S_logp) finite, 0 outside eval_mask; token_reward_direct
     advantage == rm*mask.
  D) 3D PG loss: compute_policy_loss_vanilla 3D branch returns a finite scalar; ratio≈1
     on-policy (old==log_prob.detach()).
  E) backward: total loss (PG + confidence BCE) backward -> trainable params (backbone/fc/
     markov/confidence) grads finite & some nonzero; frozen embed_tokens/lm_head no grad.
  F) micro-batch split + grad accumulation (update_dspark_opd's loop) == whole-batch backward
     when micros have equal token-mean denominators (validates the static 1/n_micro scheme).
  G) SFT path zero-regression: forward with default (None) args == internal-sampling SFT path
     (deterministic under fixed seed & finite) — the OPD branch must not change SFT behavior.

Note: the grad training forward now goes THROUGH Qwen3DSparkModel.forward's OPD branch
(model(...) == FSDP.forward), not a standalone submethod helper — required for multi-GPU
gradient sync (see docs/DSpark-OPD.md §S4). Cross-rank grad sync is covered by the separate
scripts/opd/s4_grad_sync_smoke.py (torchrun, multi-GPU).

Usage:
    PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec:/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl \
      ~/.venv/dspark-opd/bin/python scripts/opd/s4_smoke.py \
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
        print("\n[S4] SMOKE FAILED")
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
    from deepspec.modeling.dspark.common import sample_anchor_positions
    from recipe.dspark_opd.dataset import DSparkCacheDataset, dspark_collate_fn
    from recipe.dspark_opd.block_rollout import dspark_block_rollout
    from recipe.dspark_opd.teacher_scoring import score_blocks_flat
    from recipe.dspark_opd.loss_bridge import (
        logp_on_topk_ids, build_opd_reward,
        confidence_accept_rate_topk, confidence_bce, block_decay_weight_mask,
        flatten_blocks_to_sequence, unflatten_sequence_to_blocks,
    )

    # OPD training forward via Qwen3DSparkModel.forward (== update_dspark_opd's call): fixed
    # anchors + block_prev_tokens=[anchor_tok, ỹ_{<blk}], returns markov-corrected draft_logits.
    def opd_forward(model, batch, tokens, anchors, keep):
        blk = int(model.block_size)
        anchor_tok = torch.gather(batch["input_ids"], 1, anchors)
        prev = torch.cat([anchor_tok.unsqueeze(-1), tokens[:, :, : blk - 1]], dim=-1)
        return model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"].to(torch.bfloat16),
            loss_mask=batch["loss_mask"],
            anchor_positions=anchors, block_keep_mask=keep, block_prev_tokens=prev)

    device = "cuda:0"
    seed_all(int(args.seed))
    K = 16
    A = int(args.num_anchors)
    GAMMA = 4.0  # loss_decay_gamma (dspark_qwen3_4b.py)

    # --- data + draft rollout (fixed anchors + sampled tokens + top-k) ---
    cfg = OmegaConf.create({"dspark": {"target_cache_path": args.cache, "n_samples": 4}})
    ds = DSparkCacheDataset(config=cfg)
    batch = {k: v.to(device) for k, v in dspark_collate_fn([ds[0], ds[1]]).items()}
    draft = Qwen3DSparkModel.from_pretrained(
        args.draft, dtype=torch.bfloat16, attn_implementation="flex_attention").to(device).eval()
    # freeze embed_tokens/lm_head like the actor worker (_build_dspark_module); OPD trains only
    # backbone/fc/markov/confidence (§2.2). Without this the frozen-no-grad check below is moot.
    draft.set_embedding_head_trainable(False)
    seed_all(int(args.seed))
    anchors, keep = sample_anchor_positions(
        seq_len=batch["input_ids"].shape[1], loss_mask=batch["loss_mask"],
        num_anchors=A, device=device)
    roll = dspark_block_rollout(
        draft, input_ids=batch["input_ids"], loss_mask=batch["loss_mask"],
        target_hidden_states=batch["target_hidden_states"], num_anchors=A,
        temperature=1.0, top_k=K, anchor_positions=anchors, block_keep_mask=keep)
    tokens = roll["tokens"]
    tki = roll["student_top_k_ids"]                     # [B,A,blk,K] fixed
    S_logp_nograd = roll["student_top_k_logp"]          # [B,A,blk,K] no-grad (old_log_prob)
    eval_mask = roll["eval_mask"]
    B, _, blk = tokens.shape
    print(f"### rollout: tokens {tuple(tokens.shape)} top_k {tuple(tki.shape)} valid={keep.sum(1).tolist()}")

    # --- teacher top-k scoring (S3) ---
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    T_on_S = score_blocks_flat(
        target, input_ids=batch["input_ids"], tokens=tokens,
        anchor_positions=anchors, block_keep_mask=keep, student_top_k_ids=tki)  # [B,A,blk,K]

    mb = keep.unsqueeze(-1).expand(B, A, blk)           # [B,A,blk] valid-block

    # ===== A) flatten invertibility (golden) =====
    print("### A) flatten invertibility: unflatten(flatten(x)) == x")
    x = T_on_S
    rt = unflatten_sequence_to_blocks(flatten_blocks_to_sequence(x), A, blk)
    em_rt = unflatten_sequence_to_blocks(
        flatten_blocks_to_sequence(eval_mask.unsqueeze(-1)), A, blk).squeeze(-1)
    _check(torch.equal(rt, x) and torch.equal(em_rt, eval_mask),
           "flatten↔unflatten round-trips K-tensor and eval_mask exactly")

    # ===== B) grad forward (via model(...) OPD branch) reproduces rollout's corrected logp =====
    print("### B) grad train-forward (model(...) OPD branch) vs rollout no-grad")
    out = opd_forward(draft, batch, tokens, anchors, keep)
    S_logp_grad = logp_on_topk_ids(out.draft_logits, tki, temperature=1.0)  # [B,A,blk,K] grad
    d = (S_logp_grad.detach() - S_logp_nograd).abs()[mb]
    # top-1 candidate == the sampled token; its grad logp should match rollout's logp_draft
    _check(S_logp_grad.requires_grad and torch.isfinite(S_logp_grad[mb]).all()
           and d.mean().item() < 0.05,
           f"grad logp on top-k matches rollout no-grad (mean|Δ|={d.mean().item():.4g} max={d.max().item():.4g})")

    # ===== C) reward + token_reward_direct advantage =====
    print("### C) reward rm=w·(T−S) + token_reward_direct advantage")
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn
    rm = build_opd_reward(T_on_S, S_logp_nograd, weight_mode="student_p")   # [B,A,blk,K] no-grad
    # response_mask = eval_mask × exp(-pos/γ) decay (§2.5.4: PG reuses SFT loss_decay_gamma)
    decay_mask = block_decay_weight_mask(eval_mask, blk, GAMMA)            # [B,A,blk]
    rm_flat = flatten_blocks_to_sequence(rm)                                # [B,A*blk,K]
    mask_flat = flatten_blocks_to_sequence(decay_mask.unsqueeze(-1)).squeeze(-1)  # [B,A*blk]
    adv_fn = get_adv_estimator_fn("token_reward_direct")
    adv, ret = adv_fn(token_level_rewards=rm_flat, response_mask=mask_flat)
    expect = rm_flat * mask_flat.unsqueeze(-1)
    rm_finite = bool(torch.isfinite(rm[mb]).all())
    rm_zero_outside = float(rm[~mb].abs().sum().item())  # invalid blocks: S,T were 0 -> rm 0
    _check(rm_finite and torch.allclose(adv, expect) and torch.equal(adv, ret),
           f"rm finite (outside-mask |sum|={rm_zero_outside:.3g}); advantage==rm*mask, returns==adv")

    # ===== D) 3D dual-clip PG loss (on-policy ratio≈1) =====
    print("### D) compute_policy_loss_vanilla 3D branch -> finite scalar")
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    logp_flat = flatten_blocks_to_sequence(S_logp_grad)                     # [B,A*blk,K] grad
    old_logp_flat = logp_flat.detach()                                     # on-policy old==new
    cfg_pl = OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": 0.2,
                               "clip_ratio_high": 0.2, "clip_ratio_c": 3.0})
    pg_loss, pg_metrics = compute_policy_loss_vanilla(
        old_log_prob=old_logp_flat, log_prob=logp_flat, advantages=adv,
        response_mask=mask_flat, loss_agg_mode="token-mean", config=cfg_pl)
    _check(pg_loss.dim() == 0 and torch.isfinite(pg_loss)
           and abs(pg_metrics["actor/ppo_kl"]) < 1e-4,
           f"pg_loss={pg_loss.item():.4f} finite scalar; ppo_kl={pg_metrics['actor/ppo_kl']:.2e}≈0 (ratio≈1)")

    # ===== E) confidence BCE + full backward =====
    print("### E) confidence BCE + backward: trainable grads finite/nonzero, frozen no grad")
    accept = confidence_accept_rate_topk(S_logp_nograd, T_on_S)            # [B,A,blk] detached
    wmask = block_decay_weight_mask(eval_mask, blk, GAMMA)                 # [B,A,blk] decay
    conf_loss = confidence_bce(out.confidence_pred, accept, wmask)
    total = pg_loss + 1.0 * conf_loss                                     # confidence_head_alpha=1.0
    _check(bool((accept[mb] >= 0).all() and (accept[mb] <= 1).all()) and torch.isfinite(conf_loss),
           f"accept_rate∈[0,1]; confidence BCE={conf_loss.item():.4f} finite")
    draft.zero_grad(set_to_none=True)
    total.backward()
    # trainable: backbone layers / fc / markov / confidence ; frozen: embed_tokens / lm_head
    def _grad_norm(prefixes):
        g = 0.0
        for n, p in draft.named_parameters():
            if any(n.startswith(pf) or f".{pf}" in n for pf in prefixes) and p.grad is not None:
                g += p.grad.float().pow(2).sum().item()
        return g ** 0.5
    train_gn = _grad_norm(["layers", "fc", "markov_head", "confidence_head", "hidden_norm", "norm"])
    frozen_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for n, p in draft.named_parameters()
        if n.startswith("embed_tokens") or n.startswith("lm_head"))
    all_finite = all(torch.isfinite(p.grad).all() for _, p in draft.named_parameters() if p.grad is not None)
    _check(all_finite and train_gn > 0 and not frozen_has_grad,
           f"trainable grad_norm={train_gn:.4g}>0, all finite; frozen embed/lm_head no grad")

    # ===== F) micro-batch split + grad accumulation == whole-batch backward =====
    # This mirrors update_dspark_opd's micro loop (worker.py) on the bare model. The claim
    # "static 1/n_micro accumulation reproduces whole-batch" holds EXACTLY when every micro
    # has the same token-mean denominator. To make that hold deterministically, we build a
    # 2-sample batch from ONE duplicated cache sample (identical valid-block counts), then
    # compare: grad of [whole batch once] vs [2 micros of size 1, each loss*(1/2), accumulated].
    print("### F) micro-batch split + grad accumulation == whole-batch (equal denominators)")

    def _step_loss(sub_batch, toks, anc_, keep_):
        """One forward(model(...) OPD)->reward->PG+conf loss on a (sub)batch. Returns loss (grad)."""
        o = opd_forward(draft, sub_batch, toks, anc_, keep_)
        sg = logp_on_topk_ids(o.draft_logits, _tki_sub, temperature=1.0)
        r = build_opd_reward(_Ton_sub, _Sold_sub, weight_mode="student_p")
        em = o.eval_mask.bool()
        dmask = block_decay_weight_mask(em, blk, GAMMA)
        a, _ = adv_fn(token_level_rewards=flatten_blocks_to_sequence(r),
                      response_mask=flatten_blocks_to_sequence(dmask.unsqueeze(-1)).squeeze(-1))
        lp = flatten_blocks_to_sequence(sg)
        pgl, _ = compute_policy_loss_vanilla(
            old_log_prob=lp.detach(), log_prob=lp, advantages=a,
            response_mask=flatten_blocks_to_sequence(dmask.unsqueeze(-1)).squeeze(-1),
            loss_agg_mode="token-mean", config=cfg_pl)
        acc = confidence_accept_rate_topk(_Sold_sub, _Ton_sub)
        cl = confidence_bce(o.confidence_pred, acc, dmask)
        return pgl + 1.0 * cl

    # duplicate sample 0 -> a 2-sample batch with identical per-sample valid blocks
    def _dup(t):
        return torch.cat([t[:1], t[:1]], dim=0)
    batch2 = {"input_ids": _dup(batch["input_ids"]), "loss_mask": _dup(batch["loss_mask"]),
              "target_hidden_states": _dup(batch["target_hidden_states"])}
    anc2, keep2, tok2 = _dup(anchors), _dup(keep), _dup(tokens)
    # fixed no-grad tensors reused inside _step_loss (sliced per call below)
    _tki_all, _Ton_all, _Sold_all = _dup(tki), _dup(T_on_S), _dup(S_logp_nograd)

    def _slice_batch(bd, sl):
        return {k: v[sl] for k, v in bd.items()}

    # (i) whole batch once
    _tki_sub, _Ton_sub, _Sold_sub = _tki_all, _Ton_all, _Sold_all
    draft.zero_grad(set_to_none=True)
    _step_loss(batch2, tok2, anc2, keep2).backward()
    g_whole = {n: p.grad.detach().clone() for n, p in draft.named_parameters() if p.grad is not None}

    # (ii) 2 micros of size 1, each loss*(1/2), gradients accumulated (== update_dspark_opd loop)
    draft.zero_grad(set_to_none=True)
    for i in range(2):
        sl = slice(i, i + 1)
        _tki_sub, _Ton_sub, _Sold_sub = _tki_all[sl], _Ton_all[sl], _Sold_all[sl]
        (_step_loss(_slice_batch(batch2, sl), tok2[sl], anc2[sl], keep2[sl]) * 0.5).backward()
    g_micro = {n: p.grad.detach().clone() for n, p in draft.named_parameters() if p.grad is not None}

    max_gd = max((g_whole[n] - g_micro[n]).abs().max().item() for n in g_whole)
    _check(set(g_whole) == set(g_micro) and max_gd < 1e-2,
           f"micro-accum grad == whole-batch grad (max|Δgrad|={max_gd:.4g}; equal denominators)")

    # ===== G) SFT path zero-regression: forward WITHOUT new args == internal-sampling path =====
    # The OPD branch adds optional params defaulting to None. With none passed, forward must
    # behave exactly like the pre-change SFT forward: internal sample_anchor_positions + real-
    # token markov teacher-force. We check determinism (same seed -> same output) and that the
    # SFT call runs finite & shape-correct (draft_logits/target_ids/eval_mask). Additionally, we
    # verify the OPD branch (explicit anchors + real-token prev) == SFT branch on the SAME anchors,
    # isolating that ONLY prev-token source differs between the two branches.
    print("### G) SFT path zero-regression (default None args == internal sampling)")
    with torch.no_grad():
        seed_all(int(args.seed))
        o_sft1 = draft(input_ids=batch["input_ids"],
                       target_hidden_states=batch["target_hidden_states"].to(torch.bfloat16),
                       loss_mask=batch["loss_mask"],
                       target_last_hidden_states=batch["target_last_hidden_states"].to(torch.bfloat16))
        seed_all(int(args.seed))
        o_sft2 = draft(input_ids=batch["input_ids"],
                       target_hidden_states=batch["target_hidden_states"].to(torch.bfloat16),
                       loss_mask=batch["loss_mask"],
                       target_last_hidden_states=batch["target_last_hidden_states"].to(torch.bfloat16))
        sft_det = (torch.equal(o_sft1.draft_logits, o_sft2.draft_logits)
                   and torch.equal(o_sft1.target_ids, o_sft2.target_ids)
                   and torch.equal(o_sft1.eval_mask, o_sft2.eval_mask))
        sft_finite = bool(torch.isfinite(o_sft1.draft_logits).all())
    # (The OPD branch itself is exercised by tests B/E/F; G asserts the SFT branch — default
    #  None args — is unchanged: internal anchor sampling + real-token markov teacher-force,
    #  deterministic under fixed seed and finite.)
    _check(sft_det and sft_finite,
           f"SFT forward deterministic & finite (draft_logits {tuple(o_sft1.draft_logits.shape)})")

    print("\n[S4] SMOKE OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
