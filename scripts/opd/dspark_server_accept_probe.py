#!/usr/bin/env python3
"""Stage-M1 server-side accept-length probe,口径对齐本 repo eval.py。

给 upstream DSPARK server 喂【与本 repo eval.py 完全相同】的 prompt:
  - 同数据集 (eval_datasets/<task>.jsonl)，同 shuffle 种子 (seed=980406)，同 max_samples;
  - chat template + enable_thinking=False (base_evaluator.py:530-540);
  - 每 sample 用原始 /generate 接口 (才能拿到 meta_info.spec_verify_ct)，但 prompt 用
    tokenizer.apply_chat_template 手动套 (等价于 eval.py 的 encode_chat_messages)。

accept length 口径与本 repo 一致:
  repo:  acceptance_length = (accepted_draft_tokens+1) 累加 / proposal_count
  sglang: completion_tokens / spec_verify_ct  (== 每次 verify 平均提交 token 数)
  两者都是"每次投机迭代平均提交的 token 数"。

用法:
  python scripts/opd/dspark_server_accept_probe.py --task gsm8k --n 32 --temperature 1.0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import requests
from transformers import AutoTokenizer


def load_task(task: str, n: int, seed: int):
    """Mirror base_evaluator.run_dataset: load jsonl, shuffle(seed), take n."""
    path = Path("eval_datasets") / f"{task}.jsonl"
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["turns"] = row["turns"][:1]
            rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=980406)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="default False, matches base_evaluator.py:538")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.target)
    rows = load_task(args.task, args.n, args.seed)
    print(f"[cfg] task={args.task} n={len(rows)} seed={args.seed} "
          f"temp={args.temperature} enable_thinking={args.enable_thinking}")

    total_completion = 0
    total_verify = 0
    per_sample = []
    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": row["turns"][0]}]
        prompt_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=args.enable_thinking,
            tokenize=True, return_tensors=None,
        )
        # normalize to a plain list[int] (some tokenizers return BatchEncoding/nested)
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        prompt_ids = [int(t) for t in prompt_ids]
        r = requests.post(f"{args.url}/generate", json={
            "input_ids": prompt_ids,
            "sampling_params": {
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                # 显式对齐 evaluate 金标准：纯 softmax multinomial，无 top_p/top_k/min_p 截断
                # （deepspec/utils/sampling.py:logits_to_probs）。sglang 默认恰好也是这些值，
                # 显式写出防 server 端默认变动导致口径漂移。
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
            },
        }, timeout=1200)
        d = r.json()
        mi = d.get("meta_info", {})
        ct = mi.get("completion_tokens")
        vc = mi.get("spec_verify_ct")
        if not ct or not vc:
            print(f"  [{i}] MISSING spec stats: completion={ct} verify={vc}")
            continue
        al = ct / vc
        total_completion += ct
        total_verify += vc
        per_sample.append({"idx": i, "completion_tokens": ct, "spec_verify_ct": vc,
                           "accept_length": al})
        if i < 5 or i % 10 == 0:
            print(f"  [{i}] completion={ct} verify={vc} accept_len={al:.3f}")

    overall = total_completion / total_verify if total_verify else 0.0
    # also report the mean-of-per-sample (matches how repo averages per-sample then... actually
    # repo sums accepted+1 over all proposals / total proposals == micro average == overall)
    micro = overall
    macro = sum(s["accept_length"] for s in per_sample) / len(per_sample) if per_sample else 0.0

    print(f"\n=== SERVER accept length (task={args.task}, n={len(per_sample)}) ===")
    print(f"  micro (Σcompletion / Σverify)  = {micro:.4f}   <- 口径对齐 repo acceptance_length")
    print(f"  macro (mean of per-sample AL)  = {macro:.4f}")
    print(f"  total_completion={total_completion} total_verify={total_verify}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "config": vars(args),
            "micro_accept_length": micro,
            "macro_accept_length": macro,
            "total_completion": total_completion,
            "total_verify": total_verify,
            "per_sample": per_sample,
        }, indent=2))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
