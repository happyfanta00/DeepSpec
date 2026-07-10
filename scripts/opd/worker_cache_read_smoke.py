#!/usr/bin/env python3
"""worker-side cache-read equivalence smoke (optimization #4).

Optimization #4 moves target_hidden_states OUT of the driver dispatch entirely: the driver
dispatches only [B] sample INDICES, and each worker re-reads its DP-shard of samples from the
shared CacheDataset in train_step. This must be numerically identical to the old path where the
driver reads the full batch and verl chunks the tensors to each rank. This smoke proves it.

Equivalence rests on two facts:
  1. CacheDataset is DETERMINISTIC: cache[i] returns the same tensors every read (mmap of an
     immutable on-disk shard). So "driver reads sample i" and "worker reads sample i" are equal.
  2. verl dispatch chunks CONTIGUOUSLY (DataProto.chunk -> TensorDict.chunk(dim=0),
     protocol.py:887). So rank r's index-shard indexes exactly the samples that rank r would have
     received had the driver chunked the full tensor batch. Requires train_batch_size % dp == 0.

Checks (dataflow; needs the real cache but no GPU / no model):
  A) OLD (driver reads full batch -> collate -> chunk to rank r)  vs
     NEW (chunk indices to rank r -> worker reads cache[idx] -> collate)
     -> rank r's tensors bit-identical on ALL keys (incl target_hidden_states), for real config.
  B) per-rank collate (pad to per-rank max-T) vs global collate then chunk: the VALID (non-pad)
     region is identical; padding differs only in extent and is masked by loss_mask. We assert the
     real-token slices are equal and that loss_mask marks the same positions.
  C) repeat: read-unique-then-repeat_interleave(n) == old driver-repeat's per-rank shard, for the
     tensors that actually feed the model.
  D) sweep (B, dp, n) incl dp=1; and B % dp != 0 raises (documents the precondition).

Usage:
    PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec:\
/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl \
      ~/.venv/dspark-opd/bin/python scripts/opd/worker_cache_read_smoke.py \
        --cache /mnt/scratch/qwen3_4b_target_cache
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "third_party", "verl")))

# Force full-tensor mode for the DRIVER-side dataset used as the reference (so its __getitem__
# returns real tensors, not indices). The worker-read path is exercised directly via cache + adapt.
os.environ["DSPARK_CACHE_READ_MODE"] = "0"

from omegaconf import OmegaConf  # noqa: E402
from verl import DataProto  # noqa: E402
from recipe.dspark_opd.dataset import (  # noqa: E402
    DSparkCacheDataset, adapt_cache_record, dspark_collate_fn)

_FAILS = 0
_KEYS = ("input_ids", "loss_mask", "target_hidden_states", "target_last_hidden_states",
         "attention_mask", "position_ids")


def _check(cond, msg):
    global _FAILS
    if not cond:
        _FAILS += 1
    print(f"  {'[OK]' if cond else '[FAIL]'} {msg}")


def _driver_full_batch_then_chunk(ds, indices, dp_size):
    """OLD path: driver reads the full batch (full-tensor __getitem__), collates (global max-T),
    wraps as DataProto, and verl chunks it contiguously to each rank."""
    feats = [ds[i] for i in indices]                     # full tensors (mode forced to 0)
    batch = dspark_collate_fn(feats)                     # global pad to batch max-T
    dp = DataProto.from_single_dict(batch)
    return dp.chunk(chunks=dp_size)                      # [rank0 shard, rank1 shard, ...]


def _worker_read_per_rank(ds, indices, dp_size):
    """NEW path: verl chunks the INDEX tensor to each rank; each rank reads cache[idx] and collates
    its OWN shard (per-rank max-T), exactly like worker train_step does in opt #4."""
    idx_dp = DataProto.from_single_dict({"sample_index": torch.tensor(indices, dtype=torch.long)})
    idx_shards = idx_dp.chunk(chunks=dp_size)
    out = []
    for sh in idx_shards:
        local = sh.batch["sample_index"].tolist()
        feats = [adapt_cache_record(ds.cache[int(i)]) for i in local]
        out.append(dspark_collate_fn(feats))
    return out


def _valid_equal(old_shard_batch, new_shard_dict, r):
    """Compare the VALID (real-token) region. old is a DataProto (global-padded then chunked); new
    is a dict (per-rank padded). Real length per row = loss_mask/attention nonzero extent; compare
    only [:, :real_len] so different padding extents don't cause spurious mismatch."""
    ob = old_shard_batch.batch
    ok = True
    B = ob["input_ids"].shape[0]
    for row in range(B):
        # real length from attention_mask (1 over real tokens)
        rl_old = int(ob["attention_mask"][row].sum())
        rl_new = int(new_shard_dict["attention_mask"][row].sum())
        if rl_old != rl_new:
            ok = False
            print(f"      rank{r} row{row}: real_len mismatch old={rl_old} new={rl_new}")
            break
        rl = rl_old
        for k in _KEYS:
            to = ob[k][row][:rl]
            tn = new_shard_dict[k][row][:rl]
            if to.shape != tn.shape or to.dtype != tn.dtype or not torch.equal(to, tn):
                ok = False
                print(f"      rank{r} row{row} key {k}: valid-region mismatch")
                break
        if not ok:
            break
    return ok


def test_A_B_real_config(ds):
    print("[A/B] worker-read per-rank == driver-read full-then-chunk (valid region), B=64 dp=8")
    B, dp_size = 64, 8
    indices = list(range(B))
    old = _driver_full_batch_then_chunk(ds, indices, dp_size)
    new = _worker_read_per_rank(ds, indices, dp_size)
    _check(len(old) == len(new) == dp_size, f"both produce {dp_size} shards")
    all_ok = all(_valid_equal(old[r], new[r], r) for r in range(dp_size))
    _check(all_ok, "ALL 8 ranks: valid-token region of every key (incl target_hidden_states) equal")
    # loss_mask marks the same positions within real length
    lm_ok = True
    for r in range(dp_size):
        ob, nb = old[r].batch, new[r]
        for row in range(ob["input_ids"].shape[0]):
            rl = int(ob["attention_mask"][row].sum())
            if not torch.equal(ob["loss_mask"][row][:rl], nb["loss_mask"][row][:rl]):
                lm_ok = False
    _check(lm_ok, "loss_mask marks identical supervised positions across paths")


def test_C_repeat(ds):
    print("[C] read-unique-then-repeat(n) == driver-repeat per-rank shard (valid region), n=4")
    B, dp_size, n = 16, 4, 4
    indices = list(range(B))
    # OLD driver-repeat: full tensors -> repeat_interleave(n) -> chunk
    feats = [ds[i] for i in indices]
    full = dspark_collate_fn(feats)
    full_dp = DataProto.from_single_dict(full).repeat(repeat_times=n, interleave=True)
    old = full_dp.chunk(chunks=dp_size)
    # NEW: chunk indices -> read unique -> collate -> repeat_interleave(n) the tensors
    idx_dp = DataProto.from_single_dict({"sample_index": torch.tensor(indices, dtype=torch.long)})
    new = []
    for sh in idx_dp.chunk(chunks=dp_size):
        local = sh.batch["sample_index"].tolist()
        bb = dspark_collate_fn([adapt_cache_record(ds.cache[int(i)]) for i in local])
        bb = {k: v.repeat_interleave(n, dim=0) for k, v in bb.items()}
        new.append(bb)
    ok = all(_valid_equal(old[r], new[r], r) for r in range(dp_size))
    _check(ok, "repeat: worker read+repeat == driver repeat+chunk (valid region, all ranks)")


def test_D_sweep_and_precondition(ds):
    print("[D] sweep (B,dp,n) incl dp=1 + precondition B%dp!=0 raises")
    for B, dp_size in [(64, 8), (16, 2), (8, 1), (32, 4)]:
        indices = list(range(B))
        old = _driver_full_batch_then_chunk(ds, indices, dp_size)
        new = _worker_read_per_rank(ds, indices, dp_size)
        ok = len(old) == len(new) and all(_valid_equal(old[r], new[r], r) for r in range(dp_size))
        _check(ok, f"B={B} dp={dp_size}: valid region equal on all ranks")
    # precondition
    raised = False
    try:
        idx_dp = DataProto.from_single_dict(
            {"sample_index": torch.tensor(list(range(10)), dtype=torch.long)})
        idx_dp.chunk(chunks=4)  # 10 % 4 != 0
    except AssertionError:
        raised = True
    _check(raised, "B=10 dp=4: index chunk raises (documents train_batch_size % dp_world == 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/mnt/scratch/qwen3_4b_target_cache")
    args = ap.parse_args()
    print("=" * 76)
    print("worker-side cache-read equivalence smoke (opt #4; real cache, no GPU)")
    print("=" * 76)
    cfg = OmegaConf.create({"dspark": {"target_cache_path": args.cache, "n_samples": 256}})
    ds = DSparkCacheDataset(config=cfg)  # DSPARK_CACHE_READ_MODE=0 forced above -> full tensors
    test_A_B_real_config(ds)
    test_C_repeat(ds)
    test_D_sweep_and_precondition(ds)
    print("=" * 76)
    if _FAILS == 0:
        print("ALL CHECKS PASSED — worker cache-read is equivalent to driver dispatch.")
        return 0
    print(f"{_FAILS} CHECK(S) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
