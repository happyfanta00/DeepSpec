#!/usr/bin/env python3
"""S7: convert a verl FSDP OPD checkpoint -> HF-style Qwen3DSparkModel checkpoint for eval.sh.

The OPD trainer saves verl-format checkpoints:
    <verl-ckpt>/actor/model_world_size_{W}_rank_{r}.pt   (state_dict per rank)
    <verl-ckpt>/actor/huggingface/config.json            (model config)
Because the actor is wrapped with ShardingStrategy.NO_SHARD, EVERY rank's model_*.pt holds the
FULL, unwrapped Qwen3DSparkModel state_dict (verified: clean keys, no _flat_param, all 64 tensors).
So conversion just loads rank 0, instantiates Qwen3DSparkModel from the config, load_state_dict,
and save_pretrained -> config.json + model.safetensors, exactly the format
scripts/eval/eval.sh's `Qwen3DSparkModel.from_pretrained(draft_name_or_path)` expects.

Smoke assertions (run automatically): reload the converted dir with from_pretrained; check
block_size, target_layer_ids, and that EVERY parameter is allclose to the source verl tensor.

Usage:
    PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec \
    ~/.venv/dspark-opd/bin/python scripts/opd/convert_ckpt.py \
        --verl-ckpt third_party/verl/checkpoints/dspark_opd/s6_run/global_step_200 \
        --out /mnt/scratch/checkpoints/deepspec/dspark_opd_qwen3_4b/step_200
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))


def _find_actor_dir(verl_ckpt: str) -> str:
    for cand in (os.path.join(verl_ckpt, "actor"), verl_ckpt):
        if glob.glob(os.path.join(cand, "model_world_size_*_rank_*.pt")):
            return cand
    raise FileNotFoundError(
        f"no model_world_size_*_rank_*.pt under {verl_ckpt}/actor or {verl_ckpt}")


def _hf_config_dir(actor_dir: str) -> str:
    # verl saves the model config under actor/huggingface/
    for cand in (os.path.join(actor_dir, "huggingface"), actor_dir):
        if os.path.exists(os.path.join(cand, "config.json")):
            return cand
    raise FileNotFoundError(f"no config.json under {actor_dir}/huggingface or {actor_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verl-ckpt", required=True, help="verl checkpoint dir, e.g. .../global_step_200")
    ap.add_argument("--out", required=True, help="output HF-style draft checkpoint dir")
    ap.add_argument("--config-from", default=None,
                    help="optional dir with a config.json to use (default: the verl ckpt's own)")
    args = ap.parse_args()

    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel

    actor_dir = _find_actor_dir(args.verl_ckpt)
    rank0 = glob.glob(os.path.join(actor_dir, "model_world_size_*_rank_0.pt"))[0]
    print(f"[convert] actor dir : {actor_dir}")
    print(f"[convert] loading rank-0 full state_dict (NO_SHARD -> each rank is complete): {os.path.basename(rank0)}")
    src_sd = torch.load(rank0, map_location="cpu", weights_only=False)
    print(f"[convert] {len(src_sd)} tensors")

    cfg_dir = args.config_from or _hf_config_dir(actor_dir)
    print(f"[convert] config from: {cfg_dir}")

    # Instantiate from config (no weights), then load the trained weights.
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(cfg_dir, trust_remote_code=False)
    model = Qwen3DSparkModel(cfg).to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(src_sd, strict=False)
    # embed_tokens/lm_head may be tied/frozen but should still be present; report anything odd.
    if missing:
        print(f"[convert] WARNING missing keys ({len(missing)}): {missing[:8]}")
    if unexpected:
        print(f"[convert] WARNING unexpected keys ({len(unexpected)}): {unexpected[:8]}")
    assert not missing and not unexpected, "state_dict key mismatch — conversion would be lossy"

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    # copy tokenizer if the eval expects it alongside (eval loads tokenizer from TARGET, so optional)
    print(f"[convert] saved HF-style checkpoint -> {args.out}")
    for f in sorted(os.listdir(args.out)):
        sz = os.path.getsize(os.path.join(args.out, f))
        print(f"           {f}  ({sz/1e6:.1f} MB)" if sz > 1e6 else f"           {f}  ({sz} B)")

    # ---------- smoke: reload + fidelity ----------
    print("\n[smoke] reload converted dir with Qwen3DSparkModel.from_pretrained")
    reloaded = Qwen3DSparkModel.from_pretrained(args.out, dtype=torch.bfloat16)
    fails = 0

    def ck(cond, msg):
        nonlocal fails
        if not cond:
            fails += 1
        print(f"  {'[OK]' if cond else '[FAIL]'} {msg}")

    ck(int(getattr(reloaded, "block_size", -1)) == 7, f"block_size==7 (got {getattr(reloaded,'block_size',None)})")
    ck(list(reloaded.target_layer_ids) == [1, 9, 17, 25, 33],
       f"target_layer_ids==[1,9,17,25,33] (got {list(reloaded.target_layer_ids)})")
    rsd = reloaded.state_dict()
    ck(set(rsd.keys()) == set(src_sd.keys()),
       f"reloaded has same {len(src_sd)} keys as source")
    max_delta, worst = 0.0, None
    for k in src_sd:
        if k not in rsd:
            continue
        d = (rsd[k].float() - src_sd[k].float()).abs().max().item()
        if d > max_delta:
            max_delta, worst = d, k
    ck(max_delta < 1e-5, f"every tensor allclose to source (max|Δ|={max_delta:.2e} at {worst})")

    print("=" * 60)
    if fails == 0:
        print(f"CONVERSION OK — eval with:\n"
              f"  edit scripts/eval/eval.sh draft_name_or_path={args.out}  (or pass it), then\n"
              f"  bash scripts/eval/eval.sh")
        return 0
    print(f"{fails} SMOKE CHECK(S) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
