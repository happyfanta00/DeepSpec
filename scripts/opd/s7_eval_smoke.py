#!/usr/bin/env python3
"""S7 smoke: verify a converted OPD checkpoint LOADS + RUNS in the eval harness (single GPU, tiny
task) and produces sane acceptance metrics. This is NOT the full eval.sh comparison (that's the
multi-GPU, ~3000-sample E2E the user runs); it just de-risks the convert->eval chain: the
Qwen3DSparkModel.from_pretrained(converted_dir) loads, the eval draft-block inference kernel runs,
and acceptance_length / accept_rate come out finite and in-range.

Reuses the SAME Qwen3DSparkEvaluator as eval.py, only with a tiny tasks list.

Usage:
    PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec \
    CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29533 RANK=0 WORLD_SIZE=1 \
    ~/.venv/dspark-opd/bin/python scripts/opd/s7_eval_smoke.py \
        --draft_name_or_path /mnt/scratch/checkpoints/deepspec/dspark_opd_qwen3_4b/step_200 \
        --target_name_or_path Qwen/Qwen3-4B --dataset gsm8k --samples 5
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft_name_or_path", required=True)
    ap.add_argument("--target_name_or_path", default="Qwen/Qwen3-4B")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--confidence_threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=980406)
    a = ap.parse_args()

    import torch
    from types import SimpleNamespace
    from transformers import AutoConfig
    from deepspec.eval.dspark import Qwen3DSparkEvaluator

    draft_config = AutoConfig.from_pretrained(a.draft_name_or_path)
    arch = draft_config.architectures[0]
    print(f"[s7-smoke] draft arch={arch}  draft={a.draft_name_or_path}")
    assert arch == "Qwen3DSparkModel", f"expected Qwen3DSparkModel, got {arch}"

    # eval.py-style args namespace; tiny single-dataset task list
    args = SimpleNamespace(
        draft_name_or_path=a.draft_name_or_path,
        target_name_or_path=a.target_name_or_path,
        temperature=a.temperature,
        confidence_threshold=a.confidence_threshold,
        seed=a.seed,
        tensorboard_dir=None,
        step=None,
        tasks=[(a.dataset, a.samples)],
        max_new_tokens=a.max_new_tokens,
    )

    ev = Qwen3DSparkEvaluator(0, args)
    print(f"[s7-smoke] evaluator built; running {a.dataset} x{a.samples} ...")
    ev.evaluate()

    fails = 0
    rows = getattr(ev, "metrics_rows", [])
    if not rows:
        print("  [FAIL] no metrics produced")
        fails += 1
    for m in rows:
        al = float(m.get("acceptance_length", float("nan")))
        vr = float(m.get("verify_rate", float("nan")))
        ar = m.get("accept_rates_by_position", [])
        ok = al == al and al >= 1.0 and 0.0 <= vr <= 1.0  # al==al: not NaN
        fails += 0 if ok else 1
        print(f"  {'[OK]' if ok else '[FAIL]'} {m.get('dataset')}: acceptance_length={al:.3f} "
              f"verify_rate={vr:.4f} accept@k={[round(float(x),3) for x in ar[:5]]}")
    ev.clean_up()
    print("=" * 60)
    if fails == 0:
        print("S7 EVAL SMOKE PASSED — converted OPD checkpoint loads & runs; metrics sane.")
        return 0
    print(f"{fails} CHECK(S) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
