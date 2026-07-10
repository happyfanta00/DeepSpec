#!/usr/bin/env python3
"""S4 multi-GPU gradient-sync smoke (★ the fix's golden check, docs/DSpark-OPD.md §S4).

Runs under `torchrun --nproc_per_node=2`. Builds the DSpark draft FSDP-wrapped exactly like
DSparkActorRolloutRefWorker (ShardingStrategy.NO_SHARD -> per-card full params + cross-rank
grad all-reduce; bf16; use_orig_params=True; embed/lm_head frozen). Each rank feeds a DIFFERENT
data slice, runs the OPD training forward THROUGH module(...) (FSDP.forward) + loss + backward
+ one SGD step, then ALL-GATHERs each rank's root-unit trainable-head PARAMS (fc / markov_head /
confidence_head / norms) and asserts they are IDENTICAL across ranks after the step.

Why it matters: multi-GPU grad sync needs TWO things, both regression-guarded by this smoke:
  (1) training forward goes through module(...) (FSDP.forward) so FSDP1 registers its
      post-backward reduction hooks — bypassing via submethods leaves root-unit heads unsynced;
  (2) NO_SHARD (not fsdp_size=1's degenerate HYBRID mesh, whose replicate-dim all-reduce is not
      wired in this FSDP1 version) so the reduction actually fires.
If either regresses, ranks fed distinct data diverge after the step -> this smoke FAILS.

Usage:
    PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec:/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl \
      torchrun --nproc_per_node=2 scripts/opd/s4_grad_sync_smoke.py \
        --draft /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest \
        --target Qwen/Qwen3-4B --cache /mnt/scratch/qwen3_4b_target_cache --num-anchors 8 --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "third_party", "verl")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--cache", default="/mnt/scratch/qwen3_4b_target_cache")
    ap.add_argument("--num-anchors", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Apply the transformers compat shim (AutoModelForVision2Seq -> ImageTextToText) BEFORE any
    # verl import that pulls transformers — same reason as recipe/dspark_opd/task_runner.py.
    import recipe.dspark_opd  # noqa: F401  (side-effect: compat shim + rollout registration)

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from transformers import AutoModelForCausalLM
    from omegaconf import OmegaConf
    from deepspec.utils import seed_all
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
    from deepspec.modeling.dspark.common import sample_anchor_positions
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn
    from recipe.dspark_opd.dataset import DSparkCacheDataset, dspark_collate_fn
    from recipe.dspark_opd.block_rollout import dspark_block_rollout
    from recipe.dspark_opd.teacher_scoring import score_blocks_flat
    from recipe.dspark_opd.loss_bridge import (
        logp_on_topk_ids, build_opd_reward, confidence_accept_rate_topk,
        confidence_bce, block_decay_weight_mask, flatten_blocks_to_sequence)
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn, compute_policy_loss_vanilla

    # ---- distributed init ----
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    K = 16
    GAMMA = 4.0
    A = int(args.num_anchors)

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    assert world >= 2, "run with --nproc_per_node>=2 to test cross-rank grad sync"

    # ---- data: give each rank a DIFFERENT sample (distinct data shard) ----
    cfg = OmegaConf.create({"dspark": {"target_cache_path": args.cache, "n_samples": 8}})
    ds = DSparkCacheDataset(config=cfg)
    feat = ds[rank % len(ds)]                                   # rank r -> sample r
    batch = {k: v.to(device) for k, v in dspark_collate_fn([feat]).items()}

    # ---- build the bare draft; do rollout + teacher scoring FIRST (no_grad) on the BARE model,
    #      BEFORE FSDP-wrapping. (Calling submethods on nested FSDP-wrapped layers before the
    #      root's own forward corrupts FSDP's lazy-init — the real worker avoids this by using
    #      the UNWRAPPED module for rollout; we mirror that by rolling out pre-wrap.) ----
    seed_all(int(args.seed))
    draft = Qwen3DSparkModel.from_pretrained(
        args.draft, dtype=torch.bfloat16, attn_implementation="flex_attention").to(device)
    draft.set_embedding_head_trainable(False)

    seed_all(int(args.seed) + rank)
    anchors, keep = sample_anchor_positions(
        seq_len=batch["input_ids"].shape[1], loss_mask=batch["loss_mask"],
        num_anchors=A, device=device)
    roll = dspark_block_rollout(
        draft, input_ids=batch["input_ids"], loss_mask=batch["loss_mask"],
        target_hidden_states=batch["target_hidden_states"], num_anchors=A,
        temperature=1.0, top_k=K, anchor_positions=anchors, block_keep_mask=keep)
    tokens, tki, S_old = roll["tokens"], roll["student_top_k_ids"], roll["student_top_k_logp"]
    eval_mask = roll["eval_mask"]

    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    T_on_S = score_blocks_flat(target, input_ids=batch["input_ids"], tokens=tokens,
                               anchor_positions=anchors, block_keep_mask=keep, student_top_k_ids=tki)
    del target
    torch.cuda.empty_cache()

    # ---- FSDP-wrap the draft EXACTLY like the actor worker: NO_SHARD (per-card full params +
    #      cross-rank grad all-reduce). This is the FIXED config; the old fsdp_size=1 HYBRID
    #      mesh would FAIL this smoke (grads not synced). device_mesh=None -> DP global group. ----
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                        buffer_dtype=torch.bfloat16)
    fsdp = FSDP(
        draft,
        auto_wrap_policy=get_fsdp_wrap_policy(module=draft, config=None, is_lora=False),
        device_id=local_rank,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        mixed_precision=mp,
        sync_module_states=True,        # broadcast rank0 weights -> identical start
        param_init_fn=init_fn,
        use_orig_params=True,
        device_mesh=None,
    )
    fsdp.train()

    # ---- OPD training step THROUGH FSDP.forward (module(...)), then backward ----
    blk = int(draft.block_size)
    anchor_tok = torch.gather(batch["input_ids"], 1, anchors)
    block_prev = torch.cat([anchor_tok.unsqueeze(-1), tokens[:, :, : blk - 1]], dim=-1)
    out = fsdp(input_ids=batch["input_ids"],
               target_hidden_states=batch["target_hidden_states"].to(torch.bfloat16),
               loss_mask=batch["loss_mask"],
               anchor_positions=anchors, block_keep_mask=keep, block_prev_tokens=block_prev)
    S_grad = logp_on_topk_ids(out.draft_logits, tki, temperature=1.0)
    rm = build_opd_reward(T_on_S, S_old, weight_mode="student_p")
    decay = block_decay_weight_mask(eval_mask.bool(), blk, GAMMA)
    mask_flat = flatten_blocks_to_sequence(decay.unsqueeze(-1)).squeeze(-1)
    adv, _ = get_adv_estimator_fn("token_reward_direct")(
        token_level_rewards=flatten_blocks_to_sequence(rm), response_mask=mask_flat)
    lp = flatten_blocks_to_sequence(S_grad)
    cfg_pl = OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": 0.2,
                               "clip_ratio_high": 0.2, "clip_ratio_c": 3.0})
    pg_loss, _ = compute_policy_loss_vanilla(
        old_log_prob=lp.detach(), log_prob=lp, advantages=adv,
        response_mask=mask_flat, loss_agg_mode="token-mean", config=cfg_pl)
    accept = confidence_accept_rate_topk(S_old, T_on_S)
    conf_loss = confidence_bce(out.confidence_pred, accept, decay)
    loss = pg_loss + 1.0 * conf_loss
    # ---- Unambiguous grad-sync check: PARAMS-AFTER-STEP must stay identical across ranks. ----
    # Both ranks start identical (sync_module_states=True). If FSDP.forward armed the cross-rank
    # gradient all-reduce, every rank steps on the SAME (averaged) gradient → params stay
    # identical. If grads were NOT synced (the bug), each rank steps on its own local gradient
    # (fed a DIFFERENT data shard) → params diverge. This sidesteps the ambiguity of reading
    # per-rank .grad views under use_orig_params/HYBRID_SHARD (which may be pre-reduction).
    opt = torch.optim.SGD([p for p in fsdp.parameters() if p.requires_grad], lr=1.0)
    fsdp.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    # Gather full params on each rank (fsdp_size=1 → each rank holds full params; use_orig_params
    # exposes original tensors) and compare rank r vs rank 0.
    # Focus on the trainable OPD heads that live in the ROOT flat-param (the ones the bug would
    # desync): fc / markov_head / confidence_head / top-level hidden_norm & norm.
    def _is_root_head(n):
        base = n.replace("_fsdp_wrapped_module.", "")
        return (base.startswith("fc.") or base.startswith("markov_head.")
                or base.startswith("confidence_head.") or base.startswith("hidden_norm.")
                or base.startswith("norm."))

    max_cross = 0.0
    n_checked = 0
    mism = []
    with FSDP.summon_full_params(fsdp, writeback=False):
        for n, p in fsdp.named_parameters():
            if not p.requires_grad or not _is_root_head(n):
                continue
            v = p.detach().float().contiguous()
            gathered = [torch.zeros_like(v) for _ in range(world)]
            dist.all_gather(gathered, v)
            d = max((gathered[r] - gathered[0]).abs().max().item() for r in range(1, world))
            max_cross = max(max_cross, d)
            n_checked += 1
            if d > 1e-4:
                mism.append((n.replace("_fsdp_wrapped_module.", ""), round(d, 5)))

    ok = (n_checked > 0) and (max_cross < 1e-4)
    if rank == 0:
        print(f"### multi-GPU grad sync (params-after-step): world={world}, "
              f"checked {n_checked} root-unit trainable head params")
        print(f"    max cross-rank |Δparam| after 1 SGD step (lr=1, distinct data/rank) "
              f"= {max_cross:.4g}  (threshold 1e-4)")
        if mism:
            print("    DIVERGED sample:", mism[:6])
        print(f"  {'[OK]' if ok else '[FAIL]'} root-unit trainable-head params identical across "
              f"ranks after step (=> FSDP.forward synced gradients)")
        print("\n[S4-gradsync] SMOKE " + ("OK" if ok else "FAILED"))

    dist.barrier()
    dist.destroy_process_group()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
