#!/usr/bin/env python3
"""Opt #5 algorithm-correctness regression: run the FULL OPD pipeline with RECOMPUTE-hidden
(teacher forward) instead of cache-hidden, and DUMP each stage's output (S1..S4) for inspection.

This mirrors the S1-S5 gated-development checks, but as a one-shot print inspection (no gate). The
pure functions (rollout / teacher scoring / reward / loss) are unchanged by #5 and already covered
by s2/s3/s4_smoke; what #5 changes is ONLY how target_hidden_states is obtained. So this script:
  S1) prints the data path: tokens-only read + teacher recompute of target_hidden_states, and
      compares recompute-hidden vs cache-hidden (direction/cosine; deep-layer norm larger by fp8).
  S2) rollout on recompute-hidden: prints tokens/top-k/logp shapes, validity, finiteness.
  S3) teacher top-k scoring: prints T_on_S shape, finiteness, a sample block's teacher-vs-draft.
  S4) reward -> advantage -> PG + confidence loss -> backward: prints each value; asserts finite,
      trainable grads nonzero, frozen params no grad.
  X)  cross-check: does feeding recompute-hidden vs cache-hidden to rollout change the SAMPLED
      tokens / greedy argmax? (expected: near-identical greedy, small logp drift — the hidden only
      differs by fp8 saturation on outlier dims.)

Usage:
    PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec:\
/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl \
      CUDA_VISIBLE_DEVICES=0 ~/.venv/dspark-opd/bin/python scripts/opd/s5_recompute_stage_dump.py \
        --draft /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest \
        --target Qwen/Qwen3-4B --cache /mnt/scratch/qwen3_4b_target_cache --num-anchors 8
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


def _ck(cond, msg):
    global _FAILS
    if not cond:
        _FAILS += 1
    print(f"  {'[OK]' if cond else '[FAIL]'} {msg}")


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
    from deepspec.data.target_cache_dataset import CacheDataset
    from recipe.dspark_opd.dataset import adapt_cache_record, adapt_tokens_only, dspark_collate_fn
    from recipe.dspark_opd.block_rollout import dspark_block_rollout
    from recipe.dspark_opd.teacher_scoring import score_blocks_flat, recompute_target_hidden_states
    from recipe.dspark_opd.loss_bridge import (
        logp_on_topk_ids, build_opd_reward, confidence_accept_rate_topk, confidence_bce,
        block_decay_weight_mask, flatten_blocks_to_sequence, unflatten_sequence_to_blocks)
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn, compute_policy_loss_vanilla

    device = "cuda:0"
    torch.cuda.set_device(0)
    seed_all(int(args.seed))
    K, A, GAMMA = 16, int(args.num_anchors), 4.0
    print("=" * 78)
    print("Opt #5 stage dump: FULL OPD pipeline on RECOMPUTE-hidden (S1..S4)")
    print("=" * 78)

    cache = CacheDataset(cache_dir=args.cache)
    lids = list(cache.target_layer_ids)
    idxs = [0, 1]

    # ---------- teacher (also used to recompute hidden) ----------
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)

    # ================= S1: data path (tokens-only read + recompute hidden) =================
    print("\n### S1) DATA: tokens-only read + teacher recompute of target_hidden_states")
    feats_tok = [adapt_tokens_only(cache.read_tokens_only(i)) for i in idxs]
    b_tok = {k: v.to(device) for k, v in dspark_collate_fn(feats_tok).items()}
    print(f"  tokens-only keys: {sorted(b_tok.keys())}  (NO hidden read)")
    print(f"  input_ids {tuple(b_tok['input_ids'].shape)}  loss_mask sum/row={b_tok['loss_mask'].sum(1).tolist()}")
    th_recomp = recompute_target_hidden_states(
        target, input_ids=b_tok["input_ids"], attention_mask=b_tok["attention_mask"],
        target_layer_ids=lids)                                   # [B,T,L*H] bf16
    print(f"  recompute target_hidden_states {tuple(th_recomp.shape)} {th_recomp.dtype}  "
          f"finite={bool(torch.isfinite(th_recomp).all())}")
    # cache-hidden (full read) for cross-check
    feats_full = [adapt_cache_record(cache[i]) for i in idxs]
    b_full = {k: v.to(device) for k, v in dspark_collate_fn(feats_full).items()}
    th_cache = b_full["target_hidden_states"].to(device)
    _ck(th_recomp.shape == th_cache.shape and torch.isfinite(th_recomp).all(),
        f"recompute hidden shape==cache {tuple(th_recomp.shape)}, all finite")
    # tokens identical between the two read paths
    _ck(torch.equal(b_tok["input_ids"], b_full["input_ids"].to(device)),
        "tokens-only read input_ids == full read input_ids (same samples)")
    # direction vs cache per layer (cos ~1, deep norms larger)
    B, T, _ = th_recomp.shape
    H = cache.hidden_size
    am = b_tok["attention_mask"]
    print("  recompute-vs-cache per layer (real tokens):")
    cos0 = None
    for li, lid in enumerate(lids):
        cs, cn, rn = [], [], []
        for bi in range(B):
            rl = int(am[bi].sum())
            cv = th_cache[bi, :rl].reshape(rl, len(lids), H)[:, li].float()
            rv = th_recomp[bi, :rl].reshape(rl, len(lids), H)[:, li].float()
            cs.append(torch.nn.functional.cosine_similarity(cv, rv, dim=-1).mean().item())
            cn.append(cv.norm(dim=-1).mean().item()); rn.append(rv.norm(dim=-1).mean().item())
        mc = sum(cs) / len(cs)
        if li == 0:
            cos0 = mc
        print(f"    layer{lid:>2}: cos={mc:.4f}  |cache|={sum(cn)/len(cn):7.2f}  |recomp|={sum(rn)/len(rn):7.2f}")
    _ck(cos0 is not None and cos0 > 0.999, f"layer{lids[0]} cos>0.999 (no fp8 saturation) — hidden aligned")

    # draft model
    draft = Qwen3DSparkModel.from_pretrained(
        args.draft, dtype=torch.bfloat16, attn_implementation="flex_attention").to(device).eval()
    draft.set_embedding_head_trainable(False)

    def run_rollout(hidden, anchors=None, keep=None):
        seed_all(int(args.seed))
        if anchors is None:
            anchors, keep = sample_anchor_positions(
                seq_len=b_tok["input_ids"].shape[1], loss_mask=b_tok["loss_mask"],
                num_anchors=A, device=device)
        r = dspark_block_rollout(
            draft, input_ids=b_tok["input_ids"], loss_mask=b_tok["loss_mask"],
            target_hidden_states=hidden.to(torch.bfloat16), num_anchors=A,
            temperature=1.0, top_k=K, anchor_positions=anchors, block_keep_mask=keep)
        return r, anchors, keep

    # ================= S2: rollout on recompute-hidden =================
    print("\n### S2) ROLLOUT on recompute-hidden")
    roll, anchors, keep = run_rollout(th_recomp)
    tokens = roll["tokens"]; tki = roll["student_top_k_ids"]
    S_logp = roll["student_top_k_logp"]; eval_mask = roll["eval_mask"]
    B, _, blk = tokens.shape
    mb = keep.unsqueeze(-1).expand(B, A, blk)
    print(f"  tokens {tuple(tokens.shape)}  top_k_ids {tuple(tki.shape)}  valid_blocks/row={keep.sum(1).tolist()}")
    print(f"  sample0 block0 top-5 ids: {tki[0,0,0,:5].tolist()}")
    print(f"  logp_draft (valid) range [{S_logp[mb].min().item():.3f}, {S_logp[mb].max().item():.3f}]")
    _ck(torch.isfinite(S_logp[mb]).all() and (S_logp[mb] <= 1e-4).all(),
        "rollout top-k logp finite and <=0 over valid blocks")
    # sampled token (multinomial @ temp=1.0) need NOT be the argmax (top-1) — sampling is random;
    # it's USUALLY in the top-K set (both from the same corrected dist) but a tail draw can miss.
    in_topk = (tki[mb] == tokens[mb].unsqueeze(-1)).any(-1)
    print(f"  sampled-token-in-topK rate = {in_topk.float().mean()*100:.1f}% (sampling@temp=1, not argmax)")
    _ck(in_topk.float().mean().item() > 0.7,
        "most sampled tokens fall in the top-K candidate set (same corrected dist)")

    # ================= S3: teacher top-k scoring =================
    print("\n### S3) TEACHER top-k scoring (score_blocks_flat)")
    T_on_S = score_blocks_flat(
        target, input_ids=b_tok["input_ids"], tokens=tokens,
        anchor_positions=anchors, block_keep_mask=keep, student_top_k_ids=tki)
    print(f"  T_on_S {tuple(T_on_S.shape)}  finite(valid)={bool(torch.isfinite(T_on_S[mb]).all())}")
    k0 = next((i for i in range(A) if bool(keep[0, i])), 0)
    print(f"  sample0 block{k0} pos0 teacher-vs-draft top-3:")
    for k in range(3):
        print(f"    id={int(tki[0,k0,0,k]):>7}  T={float(T_on_S[0,k0,0,k]):+.3f}  D={float(S_logp[0,k0,0,k]):+.3f}")
    _ck(torch.isfinite(T_on_S[mb]).all() and (T_on_S[mb] <= 1e-4).all(),
        "teacher logp finite and <=0 over valid blocks")

    # ================= S4: reward -> advantage -> loss -> backward =================
    print("\n### S4) REWARD -> ADVANTAGE -> PG + confidence LOSS -> backward")
    # flatten invertibility
    rt = unflatten_sequence_to_blocks(flatten_blocks_to_sequence(T_on_S), A, blk)
    _ck(torch.equal(rt, T_on_S), "flatten<->unflatten round-trips exactly")
    # grad forward via model(...) OPD branch
    anchor_tok = torch.gather(b_tok["input_ids"], 1, anchors)
    prev = torch.cat([anchor_tok.unsqueeze(-1), tokens[:, :, : blk - 1]], dim=-1)
    out = draft(input_ids=b_tok["input_ids"], target_hidden_states=th_recomp.to(torch.bfloat16),
                loss_mask=b_tok["loss_mask"], anchor_positions=anchors,
                block_keep_mask=keep, block_prev_tokens=prev)
    S_grad = logp_on_topk_ids(out.draft_logits, tki, temperature=1.0)
    rm = build_opd_reward(T_on_S, S_logp, weight_mode="student_p")
    decay = block_decay_weight_mask(eval_mask, blk, GAMMA)
    adv_fn = get_adv_estimator_fn("token_reward_direct")
    adv, ret = adv_fn(token_level_rewards=flatten_blocks_to_sequence(rm),
                      response_mask=flatten_blocks_to_sequence(decay.unsqueeze(-1)).squeeze(-1))
    lp = flatten_blocks_to_sequence(S_grad)
    cfg_pl = OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": 0.2,
                               "clip_ratio_high": 0.2, "clip_ratio_c": 3.0})
    pg_loss, pg_m = compute_policy_loss_vanilla(
        old_log_prob=lp.detach(), log_prob=lp, advantages=adv,
        response_mask=flatten_blocks_to_sequence(decay.unsqueeze(-1)).squeeze(-1),
        loss_agg_mode="token-mean", config=cfg_pl)
    accept = confidence_accept_rate_topk(S_logp, T_on_S)
    conf_loss = confidence_bce(out.confidence_pred, accept, decay)
    total = pg_loss + 1.0 * conf_loss
    print(f"  rm(valid) range [{rm[mb].min().item():+.3f},{rm[mb].max().item():+.3f}]  "
          f"accept(valid) range [{accept[mb].min().item():.3f},{accept[mb].max().item():.3f}]")
    print(f"  pg_loss={pg_loss.item():+.4f}  conf_loss={conf_loss.item():.4f}  total={total.item():+.4f}  "
          f"ppo_kl={pg_m['actor/ppo_kl']:.2e}")
    _ck(torch.isfinite(total) and abs(pg_m["actor/ppo_kl"]) < 1e-4,
        "total loss finite; ppo_kl≈0 (on-policy ratio≈1)")
    total.backward()
    # trainable grads nonzero, frozen no grad
    gnz = any(p.grad is not None and p.grad.abs().sum() > 0
              for n, p in draft.named_parameters() if p.requires_grad)
    frozen_ok = all(p.grad is None for n, p in draft.named_parameters()
                    if not p.requires_grad)
    _ck(gnz, "trainable params have nonzero grad after backward")
    _ck(frozen_ok, "frozen embed/lm_head have no grad")

    # ================= X: recompute-vs-cache fed rollout (greedy stability) =================
    print("\n### X) cross-check: recompute-hidden vs cache-hidden fed to rollout (same anchors)")
    roll_c, _, _ = run_rollout(th_cache, anchors=anchors, keep=keep)
    greedy_agree = (roll_c["student_top_k_ids"][..., 0][mb] == tki[..., 0][mb]).float().mean().item()
    logp_drift = (roll_c["student_top_k_logp"][mb] - S_logp[mb]).abs().mean().item()
    print(f"  greedy(top-1) agreement recompute-vs-cache = {greedy_agree*100:.1f}%   "
          f"mean|Δlogp| = {logp_drift:.4f}")
    _ck(greedy_agree > 0.90,
        f"greedy tokens stable across hidden source ({greedy_agree*100:.1f}% agree) — algorithm robust")

    print("=" * 78)
    if _FAILS == 0:
        print("ALL STAGE CHECKS PASSED — opt #5 recompute path produces a correct, sane OPD step.")
        return 0
    print(f"{_FAILS} CHECK(S) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
