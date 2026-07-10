#!/usr/bin/env python3
"""S1 smoke-test: DSparkCacheDataset reads the real target cache correctly.

Isolated (no verl/Ray) assertions on the S1 dataset + collate:
  - dataset length matches the cache manifest num_samples (capped by n_samples if set);
  - a sample has the 6 expected keys with correct dtypes/shapes;
  - target_hidden_states is bf16 (fp8 dequantized+sanitized) and finite (no NaN/Inf);
  - shapes: input_ids/loss_mask/attention_mask/position_ids [T]; hidden [T, L*H]; last [T, H];
  - dspark_collate_fn right-pads a variable-length mini-batch to (B, T_max, ...).

Usage (my smoke-test; you can also run it):
    PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec \
        ~/.venv/dspark-opd/bin/python scripts/opd/s1_smoke.py \
        --cache /mnt/scratch/qwen3_4b_target_cache --n 4

Exit 0 on all-pass; exit 1 on first failed assertion.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

# make `recipe.dspark_opd.*` importable from the vendored verl tree
_VERL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "verl")
sys.path.insert(0, os.path.abspath(_VERL))


def _check(cond: bool, msg: str) -> None:
    print(f"  {'[OK]' if cond else '[FAIL]'} {msg}")
    if not cond:
        print("\n[S1] SMOKE FAILED")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/mnt/scratch/qwen3_4b_target_cache")
    ap.add_argument("--n", type=int, default=4, help="n_samples cap for the test")
    args = ap.parse_args()

    from recipe.dspark_opd.dataset import DSparkCacheDataset, dspark_collate_fn
    from deepspec.data.target_cache_dataset import load_target_cache_manifest

    manifest = load_target_cache_manifest(os.path.abspath(args.cache))
    n_total = int(manifest["num_samples"])
    hidden_size = int(manifest["hidden_size"])
    num_layers = len(manifest["target_layer_ids"])
    print(f"### manifest: num_samples={n_total} hidden_size={hidden_size} "
          f"layers={num_layers} hidden_dtype={manifest.get('hidden_dtype')}")

    cfg = OmegaConf.create({"dspark": {"target_cache_path": args.cache, "n_samples": args.n}})

    print("### 1) dataset construct + length")
    ds = DSparkCacheDataset(config=cfg)
    _check(len(ds) == min(args.n, n_total), f"len(ds)={len(ds)} == min(n={args.n}, total={n_total})")

    print("### 2) single sample keys / dtypes / shapes / finiteness")
    s = ds[0]
    expect_keys = {"input_ids", "attention_mask", "position_ids", "loss_mask",
                   "target_hidden_states", "target_last_hidden_states"}
    _check(set(s.keys()) == expect_keys, f"keys == {sorted(expect_keys)}")
    T = int(s["input_ids"].shape[0])
    _check(s["input_ids"].dtype == torch.long, f"input_ids dtype=long (got {s['input_ids'].dtype})")
    _check(tuple(s["attention_mask"].shape) == (T,), "attention_mask [T]")
    _check(tuple(s["position_ids"].shape) == (T,), "position_ids [T]")
    _check(tuple(s["loss_mask"].shape) == (T,), "loss_mask [T]")
    ths = s["target_hidden_states"]
    tlhs = s["target_last_hidden_states"]
    _check(tuple(ths.shape) == (T, num_layers * hidden_size),
           f"target_hidden_states [T, {num_layers}*{hidden_size}={num_layers*hidden_size}] (got {tuple(ths.shape)})")
    _check(tuple(tlhs.shape) == (T, hidden_size),
           f"target_last_hidden_states [T, {hidden_size}] (got {tuple(tlhs.shape)})")
    _check(ths.dtype == torch.bfloat16 and tlhs.dtype == torch.bfloat16,
           f"hidden dtype bf16 (got {ths.dtype}/{tlhs.dtype})")
    _check(bool(torch.isfinite(ths.float()).all()) and bool(torch.isfinite(tlhs.float()).all()),
           "hidden states finite (no NaN/Inf after fp8 sanitize)")
    print(f"      sample T={T}, input_ids[:8]={s['input_ids'][:8].tolist()}, "
          f"loss_mask.sum={int(s['loss_mask'].sum())}")

    print("### 3) dspark_collate_fn right-pads a variable-length mini-batch")
    b = min(len(ds), 3)
    feats = [ds[i] for i in range(b)]
    lens = [int(f["input_ids"].shape[0]) for f in feats]
    batch = dspark_collate_fn(feats)
    t_max = max(lens)
    _check(tuple(batch["input_ids"].shape) == (b, t_max), f"collate input_ids (B={b}, T_max={t_max})")
    _check(tuple(batch["target_hidden_states"].shape) == (b, t_max, num_layers * hidden_size),
           f"collate target_hidden_states (B, T_max, {num_layers*hidden_size})")
    # attention_mask marks real tokens: row i has exactly lens[i] ones
    am_rows_ok = all(int(batch["attention_mask"][i].sum()) == lens[i] for i in range(b))
    _check(am_rows_ok, f"attention_mask row sums == real lengths {lens}")
    _check(bool(torch.isfinite(batch["target_hidden_states"].float()).all()),
           "batched hidden finite")

    print("\n[S1] SMOKE OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
