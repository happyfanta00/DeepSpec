"""DSpark-OPD dataset (IP-3) — S1: real target-cache reading.

Wraps DeepSpec's `deepspec.data.target_cache_dataset.CacheDataset` to feed the verl
dataflow with real cached samples from `/mnt/scratch/qwen3_4b_target_cache`:
    input_ids, loss_mask, target_hidden_states, target_last_hidden_states
CacheDataset.__getitem__ already handles float8 (float8_e4m3fn) dequant + sanitize to
bf16 internally, so we just adapt its output to verl's expectations.

verl specifics handled here:
  - constructor contract: (data_files, tokenizer, processor, config, max_samples)
  - samples are VARIABLE length; verl's default collate_fn uses torch.stack (needs equal
    shapes). So this module also provides `dspark_collate_fn`, a right-padding collate
    (mirrors DeepSpec CacheCollator) wired into the trainer by task_runner.py.
  - verl's gen path pops input_ids/attention_mask/position_ids -> we add attention_mask
    (ones over real length) and position_ids (arange); input_ids int32 -> long.

Config (under `data.dspark`):
    target_cache_path : path to the target cache dir (required for real reading)
    n_samples         : optional cap for quick runs (<=0 or absent -> full dataset)

HIDDEN modes (DSPARK_HIDDEN_MODE env — docs/opd/worker-side-cache-read-design.md):
  - "recompute" (DEFAULT, opt #5): STANDARD dispatch — the driver dataloader reads the TOKENS
    (input_ids/loss_mask/attention_mask/position_ids; via read_tokens_only, ~KB/sample, prefetched
    by num_workers) and dispatches them like any verl dataset. It does NOT read/dispatch the large
    target_hidden_states; the worker RE-RUNS the co-resident teacher to recompute it (a ~0.1s GPU
    prefill forward). Fastest & most stable, and matches inference (live-teacher hidden, bf16).
  - "cache" (opt #4): dispatch only the sample index; the worker re-reads the FULL sample (incl.
    target_hidden_states) from the shared mmap'd cache. No hidden dispatch, but pays the cache
    cold-read on the worker's critical path (no dataloader prefetch overlap).
  - "dispatch" (opt #3): the driver reads full tensors (tokens + target_hidden_states) and
    dispatches them (hidden serialized through the Ray object store). Legacy; kept for A/B.
  Only "cache" is index-only. "recompute"/"dispatch" dispatch real driver-read tensors (recompute:
  tokens; dispatch: tokens+hidden). The cache is deterministic, so all three are equivalent.
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset

from deepspec.data.target_cache_dataset import CacheDataset

# Data path mode. Default "recompute" (#5). Legacy env DSPARK_CACHE_READ_MODE=0 forces "dispatch".
_HIDDEN_MODE = os.environ.get(
    "DSPARK_HIDDEN_MODE",
    "dispatch" if os.environ.get("DSPARK_CACHE_READ_MODE") == "0" else "recompute",
)
assert _HIDDEN_MODE in ("recompute", "cache", "dispatch"), f"bad DSPARK_HIDDEN_MODE={_HIDDEN_MODE}"


def adapt_tokens_only(rec: dict) -> dict:
    """CacheDataset record (input_ids/loss_mask) -> verl-shaped sample dict WITHOUT hidden (opt #5).

    The gen-path keys (input_ids/attention_mask/position_ids) + loss_mask; the worker recomputes
    target_hidden_states via the teacher. adapt_cache_record extends this with the cached hidden.
    """
    input_ids = rec["input_ids"].to(torch.long)          # int32 -> long
    seq_len = int(input_ids.shape[0])
    return {
        # gen-path keys (verl pops these three when going through its gen pipeline)
        "input_ids": input_ids,
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "position_ids": torch.arange(seq_len, dtype=torch.long),
        "loss_mask": rec["loss_mask"].to(torch.long),
    }


def adapt_cache_record(rec: dict) -> dict:
    """CacheDataset record -> verl-shaped sample dict (tokens via adapt_tokens_only + cached hidden).

    Shared by the driver dataset (full-tensor mode) and the worker-side read path so both
    produce IDENTICAL sample dicts for the same cache index.
    """
    sample = adapt_tokens_only(rec)
    # DSpark cached tensors (bf16 after CacheDataset's fp8 sanitize)
    sample["target_hidden_states"] = rec["target_hidden_states"]
    sample["target_last_hidden_states"] = rec["target_last_hidden_states"]
    return sample


class DSparkCacheDataset(Dataset):
    """Real target-cache dataset for DSpark-OPD (S1)."""

    def __init__(self, data_files=None, tokenizer=None, processor=None, config=None,
                 max_samples: int = -1):
        self.tokenizer = tokenizer
        self.config = config
        d = {}
        try:
            d = dict(config.get("dspark", {})) if config is not None else {}
        except Exception:  # noqa: BLE001
            d = {}
        cache_path = d.get("target_cache_path", None)
        if not cache_path:
            raise ValueError(
                "data.dspark.target_cache_path is required for DSparkCacheDataset (S1). "
                "Set it to e.g. /mnt/scratch/qwen3_4b_target_cache"
            )
        self.cache = CacheDataset(cache_dir=str(cache_path))
        self.n_samples = len(self.cache)
        cap = int(d.get("n_samples", -1) or -1)
        if cap > 0:
            self.n_samples = min(self.n_samples, cap)
        if max_samples and max_samples > 0:
            self.n_samples = min(self.n_samples, int(max_samples))
        print(
            f"[DSparkCacheDataset:S1] cache={cache_path} "
            f"len={len(self.cache)} using={self.n_samples} "
            f"hidden_dtype={self.cache.hidden_dtype} "
            f"layers={self.cache.num_target_layers}x{self.cache.hidden_size}"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> dict:
        if _HIDDEN_MODE == "cache":
            # #4: dispatch only the index; the worker re-reads the full sample from the mmap'd cache.
            return {"sample_index": torch.tensor(int(index), dtype=torch.long)}
        if _HIDDEN_MODE == "recompute":
            # #5 (standard dispatch): the driver reads TOKENS only (~KB, prefetched by num_workers)
            # and dispatches them; the worker recomputes target_hidden_states via the teacher. No
            # large hidden read/dispatch here.
            return adapt_tokens_only(self.cache.read_tokens_only(int(index)))
        # #3 (legacy): driver reads the full sample incl. target_hidden_states and dispatches it.
        return adapt_cache_record(self.cache[int(index)])


def dspark_collate_fn(features: list[dict]) -> dict:
    """Right-pad variable-length samples into a batch (verl's default stack can't).

    Index-only mode (opt #4): features are {"sample_index": scalar long} -> {"sample_index": [B]}.
    Full mode: 1-D keys (input_ids/attention_mask/position_ids/loss_mask) -> (B, T_max) padded 0;
    2-D hidden keys -> (B, T_max, H) padded 0. attention_mask marks real tokens.
    Tokens-only (opt #5): the 2-D hidden keys are absent (worker recomputes hidden) — collate skips
    any key not present in the features.
    """
    if "sample_index" in features[0]:
        return {"sample_index": torch.stack([f["sample_index"] for f in features], dim=0)}

    keys_1d = ("input_ids", "attention_mask", "position_ids", "loss_mask")
    keys_2d = ("target_hidden_states", "target_last_hidden_states")
    bsz = len(features)
    t_max = max(int(f["input_ids"].shape[0]) for f in features)

    batch: dict = {}
    for key in keys_1d:
        if key not in features[0]:
            continue
        dtype = features[0][key].dtype
        out = torch.zeros((bsz, t_max), dtype=dtype)
        for i, f in enumerate(features):
            n = int(f[key].shape[0])
            out[i, :n] = f[key]
        batch[key] = out
    for key in keys_2d:
        if key not in features[0]:
            continue
        h = int(features[0][key].shape[1])
        dtype = features[0][key].dtype
        out = torch.zeros((bsz, t_max, h), dtype=dtype)
        for i, f in enumerate(features):
            n = int(f[key].shape[0])
            out[i, :n] = f[key]
        batch[key] = out
    return batch
