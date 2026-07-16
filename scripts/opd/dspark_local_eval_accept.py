#!/usr/bin/env python3
"""Stage-M1 golden 基准：本 repo eval.py 的 evaluator 在 gsm8k 上的 accept length。

薄 wrapper——不改 eval.py 源码，直接构造 Qwen3DSparkEvaluator，跑单 GPU、限定
样本数，拿到本地 HF 前向的 acceptance_length。与 server probe 使用【完全相同】的
prompt 集（同 dataset + 同 shuffle seed 980406 + 同 enable_thinking=False，见
base_evaluator.run_dataset），从而可直接对拍。

单 GPU (world_size=1) 时 run_dataset 的 shuffle(seed)[:max_samples] + rank::world
分片 == 前 max_samples 条，与 dspark_server_accept_probe.py 的 shuffle(seed)[:n] 一致。

用法（在 server 没占用的 GPU 上，如 GPU1）：
  CUDA_VISIBLE_DEVICES=1 python scripts/opd/dspark_local_eval_accept.py \
      --target Qwen/Qwen3-4B --draft <ckpt> --task gsm8k --n 16 --temperature 1.0
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=980406)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)  # 对齐 eval.py:36 金标准
    args = ap.parse_args()

    # Build the args namespace the evaluator expects (mirror eval.py parse_args).
    ev_args = SimpleNamespace(
        target_name_or_path=args.target,
        draft_name_or_path=args.draft,
        temperature=args.temperature,
        confidence_threshold=0.0,
        tensorboard_dir=None,
        step=None,
        entropy_stats=False,
        entropy_output_dir="./entropy_stats",
        tasks=[(args.task, args.n)],  # (dataset, max_samples) — cap to n
        seed=args.seed,
    )
    # set generously; base_evaluator reads args.max_new_tokens via generate_one_sample
    setattr(ev_args, "max_new_tokens", args.max_new_tokens)

    # init_dist needs torchrun-style env; single process → world_size=1 so the
    # dist.all_reduce inside allreduce_response_metrics is a no-op reduction.
    import os
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")

    from deepspec.eval.dspark.evaluator import Qwen3DSparkEvaluator

    # local_rank=0, single process (world_size=1 via CUDA_VISIBLE_DEVICES=1 GPU)
    evaluator = Qwen3DSparkEvaluator(0, ev_args)

    # We only need accept length; disable the confidence/entropy recorders so we
    # can call run_dataset directly without the full evaluate() lifecycle (which
    # would otherwise call recorder.start() to init dataset_metrics). Their
    # _post_verify hooks are guarded by "is not None", so None = skip cleanly.
    evaluator.confidence_head_recorder = None
    evaluator.entropy_recorder = None

    # run just the one dataset, capped to n
    responses = evaluator.run_dataset(dataset_name=args.task, max_samples=args.n)
    summary = evaluator.allreduce_response_metrics(responses)
    metrics = evaluator.build_metrics_row(dataset_name=args.task, metric_summary=summary)

    print(f"\n=== LOCAL eval.py accept length (task={args.task}, n={args.n}) ===")
    print(f"  acceptance_length         = {metrics['acceptance_length']:.4f}   <- golden 基准")
    print(f"  draft_tokens_per_proposal = {metrics['draft_tokens_per_proposal']:.4f}")
    print(f"  verify_rate               = {metrics['verify_rate']:.4f}")
    print(f"  num_samples               = {metrics['num_samples']}")
    # accept rates by position (per-slot accept prob)
    arr = metrics.get("accept_rates_by_position")
    if arr:
        shown = [f"{a:.3f}" if a is not None else "-" for a in arr]
        print(f"  accept_rate@pos           = {shown}")
    evaluator.clean_up()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
