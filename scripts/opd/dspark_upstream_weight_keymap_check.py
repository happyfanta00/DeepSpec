#!/usr/bin/env python3
"""Offline key-map check: does a DSpark HF checkpoint load cleanly into upstream
sglang's native ``Qwen3DSparkModel`` (models/dspark.py) without CUDA/dist init?

We do NOT instantiate the heavy model (QKVParallelLinear needs a TP world +
CUDA). Instead we replay the *exact* rules of upstream ``load_weights``:

  - DSparkDraftMixin.load_weights (models/dspark.py:38): route by prefix
    ``markov_head.`` / ``confidence_head.`` / backbone; skip
    _DSPARK_SKIPPED_WEIGHT_PREFIXES = (embed_tokens., lm_head., rotary_emb.).
  - Backbone DFlashDraftModel.load_weights (dflash.py:429): stacked-params
    fuse q/k/v -> qkv_proj and gate/up -> gate_up_proj; everything else direct;
    truly-unknown backbone keys are silently ignored.

and diff the ckpt key set against the parameter set upstream expects, reporting
missing (expected but absent) and unexpected (present but unroutable) keys.

Usage:
    python scripts/opd/dspark_upstream_weight_keymap_check.py \
        /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest
"""
from __future__ import annotations

import json
import os
import sys

from safetensors import safe_open

SKIPPED_PREFIXES = ("embed_tokens.", "lm_head.", "rotary_emb.")
# (fused_param, source_weight, shard) — matches dflash.py stacked_params_mapping
STACKED = [
    ("qkv_proj", "q_proj"),
    ("qkv_proj", "k_proj"),
    ("qkv_proj", "v_proj"),
    ("gate_up_proj", "gate_proj"),
    ("gate_up_proj", "up_proj"),
]


def expected_backbone_params(num_layers: int) -> set[str]:
    """The backbone parameter names upstream would expose (post-fusion)."""
    params: set[str] = {"norm.weight", "fc.weight", "hidden_norm.weight"}
    per_layer = [
        "self_attn.qkv_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]
    for i in range(num_layers):
        for p in per_layer:
            params.add(f"layers.{i}.{p}")
    return params


def expected_head_params(cfg: dict) -> set[str]:
    params: set[str] = set()
    if int(cfg.get("markov_rank", 0)) > 0:
        params.add("markov_head.markov_w1.weight")
        params.add("markov_head.markov_w2.weight")
        if str(cfg.get("markov_head_type", "vanilla")).lower() == "gated":
            params.add("markov_head.gate_proj.weight")
            params.add("markov_head.gate_proj.bias")
        elif str(cfg.get("markov_head_type", "vanilla")).lower() == "rnn":
            params.add("markov_head.joint_proj.weight")
            params.add("markov_head.joint_proj.bias")
    if cfg.get("enable_confidence_head"):
        params.add("confidence_head.proj.weight")
        params.add("confidence_head.proj.bias")
    return params


def route(name: str) -> tuple[str, str | None]:
    """Return (bucket, resolved_param_name-or-None) mirroring upstream routing."""
    if any(name.startswith(p) for p in SKIPPED_PREFIXES):
        return "skipped", None
    if name.startswith("markov_head."):
        return "markov", name
    if name.startswith("confidence_head."):
        return "confidence", name
    # backbone: try stacked fusion first
    for fused, src in STACKED:
        if f".{src}." in name:
            return "backbone", name.replace(src, fused)
    return "backbone", name  # direct


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    ckpt = sys.argv[1]
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    with safe_open(os.path.join(ckpt, "model.safetensors"), "pt") as h:
        ckpt_keys = list(h.keys())

    num_layers = int(cfg["num_hidden_layers"])
    expected = expected_backbone_params(num_layers) | expected_head_params(cfg)

    resolved: set[str] = set()
    skipped: list[str] = []
    unroutable: list[str] = []
    buckets = {"backbone": 0, "markov": 0, "confidence": 0, "skipped": 0}
    for name in ckpt_keys:
        bucket, target = route(name)
        buckets[bucket] += 1
        if bucket == "skipped":
            skipped.append(name)
            continue
        if target in expected:
            resolved.add(target)
        else:
            unroutable.append(f"{name} -> {target}")

    missing = sorted(expected - resolved)

    print(f"ckpt: {ckpt}")
    print(f"config: markov_rank={cfg.get('markov_rank')} "
          f"markov_head_type={cfg.get('markov_head_type')} "
          f"enable_confidence_head={cfg.get('enable_confidence_head')} "
          f"confidence_head_with_markov={cfg.get('confidence_head_with_markov')} "
          f"num_hidden_layers={num_layers} block_size={cfg.get('block_size')}")
    print(f"ckpt keys: {len(ckpt_keys)} | expected params: {len(expected)}")
    print(f"routed: {buckets} | skipped(embed/lm_head): {skipped}")
    print(f"resolved expected params: {len(resolved)}/{len(expected)}")

    ok = True
    if missing:
        ok = False
        print(f"\nMISSING (expected but not in ckpt) [{len(missing)}]:")
        for m in missing:
            print("  ", m)
    if unroutable:
        ok = False
        print(f"\nUNEXPECTED/UNROUTABLE ckpt keys [{len(unroutable)}]:")
        for u in unroutable:
            print("  ", u)

    print("\nRESULT:", "CLEAN LOAD ✅" if ok else "MISMATCH ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
