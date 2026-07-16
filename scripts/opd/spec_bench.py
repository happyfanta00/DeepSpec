#!/usr/bin/env python3
"""DSpark-on-SGLang 投机解码性能 benchmark（Stage-1a 及后续阶段复用）。

打一个已启动的 SGLang server 的 HTTP /generate endpoint，测吞吐/延迟；若 server 以投机
解码模式启动（DFLASH / DSPARK），额外统计 accept length。脚本本身不关心 server 是什么模式
——baseline vs spec 由你启动 server 的命令决定，两次跑同一脚本、对比输出即可判断投机解码是否
正常（无损 + 有加速）。

用法（两次跑，server 分别以 baseline / spec 启动）：
    # A) baseline：纯 serving
    $PY -m sglang.launch_server --model-path Qwen/Qwen3-4B --port 30000
    $PY scripts/opd/spec_bench.py --port 30000 --n 128 --tag baseline \
        --out docs/opd/bench/stage1a_baseline.json

    # B) spec：投机解码（DFlash 候选1）
    $PY -m sglang.launch_server --model-path Qwen/Qwen3-4B \
        --speculative-algorithm DFLASH \
        --speculative-draft-model-path bingyang-lei/Qwen3-4B-Ins-Draft-OPD \
        --speculative-dflash-block-size 7 --port 30000
    $PY scripts/opd/spec_bench.py --port 30000 --n 128 --tag dflash \
        --out docs/opd/bench/stage1a_dflash.json

    # 对比两份结果
    $PY scripts/opd/spec_bench.py --compare \
        docs/opd/bench/stage1a_baseline.json docs/opd/bench/stage1a_dflash.json

评测口径对齐 evaluate 金标准（eval.py:36-37, base_evaluator.py:538）：默认 temperature=1.0
（走 rejection sampling 路径，非 greedy argmax）、max_new_tokens=2048、套 chat template +
enable_thinking=False、固定 seed 980406 抽样可复现。加速比 = spec 吞吐 / baseline 吞吐。
依赖 transformers（套 chat template）+ stdlib urllib；`--no-chat-template` 可退回裸文本(不推荐)。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import time
import urllib.request
import urllib.error
from pathlib import Path

DEFAULT_DATASET = "eval_datasets/perfectblend.jsonl"


# ------------------------------ 数据子集（可复现） ------------------------------
def load_prompts(dataset: str, n: int, seed: int, max_chars: int) -> list[str]:
    """固定 seed 从 dataset 抽 n 条单轮 prompt（过滤过长/多轮），保证跨运行可复现。"""
    rows: list[str] = []
    with open(dataset) as f:
        for line in f:
            try:
                turns = json.loads(line)["turns"]
            except Exception:
                continue
            if not turns or len(turns) != 1:
                continue  # 只取单轮，避免多轮上下文干扰
            p = turns[0]
            if isinstance(p, str) and 0 < len(p) <= max_chars:
                rows.append(p)
    rng = random.Random(seed)
    rng.shuffle(rows)
    picked = rows[:n]
    if len(picked) < n:
        print(f"[warn] 仅得到 {len(picked)}/{n} 条符合条件的 prompt")
    return picked


# ------------------------------ 单请求 ------------------------------
def gen_once(base_url: str, prompt, max_new_tokens: int, temperature: float,
             timeout: float) -> dict:
    """发一个 /generate 请求，返回 {text, latency, completion_tokens, spec_verify_ct}。

    prompt 可以是 str（裸文本，走 /generate 的 "text"）或 list[int]（已套 chat
    template 的 token ids，走 "input_ids"）——后者与本 repo eval / 接受率对拍口径一致。
    """
    # 显式对齐 evaluate 金标准：纯 softmax multinomial，无 top_p/top_k/min_p 截断
    # （deepspec/utils/sampling.py:logits_to_probs）。显式写出防 server 端默认漂移。
    payload = {"sampling_params": {
        "temperature": temperature, "max_new_tokens": max_new_tokens,
        "top_p": 1.0, "top_k": -1, "min_p": 0.0,
    }}
    if isinstance(prompt, str):
        payload["text"] = prompt
    else:
        payload["input_ids"] = list(prompt)
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode())
    latency = time.perf_counter() - t0
    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    return {
        "text": out.get("text", "") if isinstance(out, dict) else "",
        "latency": latency,
        "completion_tokens": meta.get("completion_tokens"),
        "spec_verify_ct": meta.get("spec_verify_ct"),
    }


# ------------------------------ 就绪探测 ------------------------------
def wait_ready(base_url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def flush_cache(base_url: str) -> None:
    try:
        urllib.request.urlopen(f"{base_url}/flush_cache", timeout=10).read()
    except Exception:
        pass


# ------------------------------ benchmark 主流程 ------------------------------
def run_bench(args) -> dict:
    base_url = f"http://{args.host}:{args.port}"
    if not wait_ready(base_url, args.ready_timeout):
        raise SystemExit(f"[error] server {base_url} 未就绪（/health 超时 {args.ready_timeout}s）")

    prompts = load_prompts(args.dataset, args.n, args.seed, args.max_prompt_chars)

    # 套 chat template（默认开）：与本 repo eval / 接受率对拍口径一致
    # （base_evaluator.py:530-540 的 apply_chat_template + enable_thinking=False）。
    # 转成 token ids 后 gen_once 走 /generate 的 "input_ids"。
    if not args.no_chat_template:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.target)
        encoded = []
        for text in prompts:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, enable_thinking=args.enable_thinking,
                tokenize=True, return_tensors=None,
            )
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            encoded.append([int(t) for t in ids])
        prompts = encoded
        tmpl = f"chat_template(enable_thinking={args.enable_thinking})"
    else:
        tmpl = "raw_text"

    print(f"[info] {len(prompts)} prompts | tag={args.tag} | concurrency={args.concurrency} "
          f"| max_new_tokens={args.max_new_tokens} | temp={args.temperature} | prompt={tmpl}")

    # 预热（不计入统计）：跑几条 + flush，稳定 CUDA graph / 缓存
    for p in prompts[:min(args.warmup, len(prompts))]:
        try:
            gen_once(base_url, p, args.max_new_tokens, args.temperature, args.timeout)
        except Exception as e:
            print(f"[warn] warmup 失败: {e!r}")
    flush_cache(base_url)

    results: list[dict] = []
    wall0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(gen_once, base_url, p, args.max_new_tokens,
                          args.temperature, args.timeout): i
                for i, p in enumerate(prompts)}
        for fut in cf.as_completed(futs):
            i = futs[fut]
            try:
                r = fut.result(); r["idx"] = i; results.append(r)
            except Exception as e:
                print(f"[warn] req#{i} 失败: {e!r}")
    wall = time.perf_counter() - wall0

    ok = [r for r in results if r.get("completion_tokens")]
    n_ok = len(ok)
    total_out = sum(r["completion_tokens"] for r in ok)
    lat = sorted(r["latency"] for r in ok)

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else None

    # accept length（仅投机解码模式有意义）：每 verify 步平均接受的 draft token 数
    #   completion_tokens ≈ Σ 各 verify 步提交的 token；spec_verify_ct = verify 步数
    #   accept_length = completion_tokens / spec_verify_ct - 1（减 1 去掉每步必有的 bonus/修正）
    spec = [r for r in ok if r.get("spec_verify_ct")]
    accept_lengths = [r["completion_tokens"] / r["spec_verify_ct"] - 1.0
                      for r in spec if r["spec_verify_ct"] > 0]
    summary = {
        "tag": args.tag,
        "n_requested": args.n,
        "n_ok": n_ok,
        "wall_s": round(wall, 3),
        "throughput_tok_s": round(total_out / wall, 2) if wall > 0 else None,
        "throughput_req_s": round(n_ok / wall, 3) if wall > 0 else None,
        "total_output_tokens": total_out,
        "latency_s": {
            "mean": round(sum(lat) / n_ok, 3) if n_ok else None,
            "p50": round(pct(lat, 0.5), 3) if lat else None,
            "p90": round(pct(lat, 0.9), 3) if lat else None,
            "p99": round(pct(lat, 0.99), 3) if lat else None,
        },
        "spec": {
            "is_speculative": bool(accept_lengths),
            "mean_accept_length": round(sum(accept_lengths) / len(accept_lengths), 3)
                                  if accept_lengths else None,
            "n_spec_reqs": len(accept_lengths),
        },
        "config": {
            "dataset": args.dataset, "seed": args.seed, "concurrency": args.concurrency,
            "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
            "max_prompt_chars": args.max_prompt_chars,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary}
        if args.dump_outputs:
            payload["outputs"] = [{"idx": r["idx"], "text": r["text"]}
                                  for r in sorted(ok, key=lambda x: x["idx"])]
        outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"[info] 结果写入 {outp}")
    return summary


# ------------------------------ 对比两份结果 ------------------------------
def compare(baseline_path: str, spec_path: str) -> None:
    b = json.loads(Path(baseline_path).read_text())
    s = json.loads(Path(spec_path).read_text())
    bs, ss = b["summary"], s["summary"]
    bt, st = bs["throughput_tok_s"], ss["throughput_tok_s"]
    print("=" * 60)
    print(f"  baseline [{bs['tag']}]  vs  spec [{ss['tag']}]")
    print("=" * 60)
    print(f"  speedup          : x{round(st / bt, 2) if bt else '?'}   "
          f"(throughput {bt} -> {st} tok/s)")
    print(f"  latency p50 (s)  : {bs['latency_s']['p50']}  ->  {ss['latency_s']['p50']}")
    print(f"  latency mean (s) : {bs['latency_s']['mean']}  ->  {ss['latency_s']['mean']}")
    # accept len：内部 mean_accept_length = completion/verify − 1（去 bonus）；
    # 换算成 evaluate 金标准口径 completion/verify（含 bonus）+1 后再报，便于与 eval 对齐。
    mal = ss["spec"]["mean_accept_length"]
    if mal is not None:
        print(f"  accept len       : {round(mal + 1, 3)}  (=completion/verify，含 bonus，对齐 evaluate 口径)")
    print("=" * 60)
    print("  判据：spec 吞吐 > baseline（加速比 >1）、accept len 与 evaluate 同口径量级一致。")
    print("  注：无损性由内核级 rejection 等价 + 分布对拍坐实，不用 greedy 逐字 diff（temp=1.0 采样下无意义）。")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SGLang 投机解码 benchmark（Stage-1a+）")
    p.add_argument("--compare", nargs=2, metavar=("BASELINE_JSON", "SPEC_JSON"),
                   help="对比两份结果 json（不发请求）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--n", type=int, default=128, help="benchmark prompt 条数")
    p.add_argument("--seed", type=int, default=980406, help="抽样种子（复现子集）")
    p.add_argument("--max-prompt-chars", type=int, default=2000, help="过滤过长 prompt")
    p.add_argument("--concurrency", type=int, default=16)
    # 默认对齐 evaluate 金标准（eval.py:36-37, base_evaluator.py:538）：
    # temperature=1.0（走 rejection sampling 路径，非 greedy argmax）、max_new_tokens=2048、
    # 套 chat template + enable_thinking=False。评测口径务必与 evaluate 一致。
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="默认 1.0，对齐 eval.py 金标准（走 rejection sampling）")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--timeout", type=float, default=600.0, help="单请求超时(s)")
    p.add_argument("--ready-timeout", type=float, default=600.0, help="等 server 就绪超时(s)")
    p.add_argument("--target", default="Qwen/Qwen3-4B",
                   help="tokenizer（套 chat template 用；须与 server target 一致）")
    p.add_argument("--no-chat-template", action="store_true",
                   help="不套 chat template、发裸文本（默认套，与 eval 口径一致）")
    p.add_argument("--enable-thinking", action="store_true",
                   help="chat template enable_thinking（默认 False，与 base_evaluator 一致）")
    p.add_argument("--tag", default="run", help="标签（baseline / dflash / dspark）")
    p.add_argument("--out", default=None, help="结果 json 输出路径")
    p.add_argument("--dump-outputs", action="store_true",
                   help="连同各请求输出文本一起落盘（供无损性 diff）")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.compare:
        compare(args.compare[0], args.compare[1])
        return
    run_bench(args)


if __name__ == "__main__":
    main()
