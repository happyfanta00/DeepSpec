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

--check-dspark-stream 附加校验 (Stage-M2 T1):
  在完全相同的对齐口径下,额外抽 meta_info["dspark_accept_state"] (T1 生产的单条
  per-token accept/reject 二态流, 0=ACCEPT / 1=COMMIT_BOUNDARY), 逐样本断言自洽:
    - #(state==1)          == spec_verify_ct           (每轮恰一个 boundary)
    - len(accept_state)    == completion_tokens - 1     (prefill 首 token 偏移,见 T4)
                              或 == completion_tokens   (容忍未来 sglang 吐全长)
    - (len+首token)/verify ≈ spec_accept_length         (重建 accept_len 精确吻合)

--viz 额外把 dspark_accept_state 可视化（终端 ANSI，非 tty 自动关色）：
  - 每样本一条彩色 strip: ▶=seed  •=accept  |=boundary（逐 token 从左到右读，| 收尾一个 block）
  - 聚合 per-slot 接受率条形图（应逐 slot 单调递减，对齐 §4.2 golden [0.937,0.874,...]）
  - block 长度直方图（每轮提交 token 数，full block == γ+1）
  --viz 隐含 --check-dspark-stream；--out 时附带落盘原始 streams 供离线重绘。

用法:
  python scripts/opd/dspark_server_accept_probe.py --task gsm8k --n 32 --temperature 1.0
  python scripts/opd/dspark_server_accept_probe.py --task gsm8k --n 8 --check-dspark-stream
  python scripts/opd/dspark_server_accept_probe.py --task gsm8k --n 8 --viz
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import requests
from transformers import AutoTokenizer

# ANSI colors, auto-disabled when not a tty (e.g. piped to a file).
_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


# accept/reject stream codes (mirror upstream batch_result_processor.py)
ACCEPT, BOUNDARY, SEED = 0, 1, 2
# glyph + color per code, for the per-sample strip
_GLYPH = {
    ACCEPT: ("32", "•"),    # green dot: drafted token accepted
    BOUNDARY: ("31", "|"),  # red bar: verify-round boundary (bonus/correction)
    SEED: ("36", "▶"),      # cyan: prefill seed (block-0 anchor)
}


def render_stream_strip(st: list[int], width: int = 100) -> str:
    """One colored glyph per output token: ▶ seed, • accept, | boundary.

    Truncated to ``width`` glyphs with an ellipsis so long responses stay on one
    screen. Reads left-to-right as the response; each ``|`` closes a verify block.
    """
    glyphs = []
    shown = st[:width]
    for s in shown:
        code, ch = _GLYPH.get(s, ("0", "?"))
        glyphs.append(_c(code, ch))
    tail = f" …(+{len(st) - width})" if len(st) > width else ""
    return "".join(glyphs) + tail


def per_slot_accept_rate(streams: list[list[int]], gamma: int) -> list[float]:
    """Aggregate accept rate at each in-block slot across all samples.

    Splits each stream into verify blocks (a block runs from an anchor —
    SEED or the token after a BOUNDARY — up to and including the next BOUNDARY).
    Slot i (0-indexed, i < gamma) is "accepted" when the block reached position
    i as an ACCEPT rather than closing (BOUNDARY) at or before it. Mirrors the
    golden per-position accept rate in design §4.2 (monotone decreasing = healthy).
    """
    reached = [0] * gamma   # blocks that had a token proposed at slot i
    accepted = [0] * gamma  # blocks whose slot i was an ACCEPT
    for st in streams:
        slot = 0
        for s in st:
            if s == SEED:
                # seed is block 0's anchor, not an in-block draft slot
                slot = 0
                continue
            if slot < gamma:
                reached[slot] += 1
                if s == ACCEPT:
                    accepted[slot] += 1
            if s == BOUNDARY:
                slot = 0  # next token starts a fresh block
            else:
                slot += 1
    return [accepted[i] / reached[i] if reached[i] else 0.0 for i in range(gamma)]


def render_bar(frac: float, width: int = 30) -> str:
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def block_length_histogram(streams: list[list[int]]) -> dict[int, int]:
    """Count verify blocks by committed length (accepts + 1 bonus; 1..gamma+1)."""
    hist: dict[int, int] = {}
    for st in streams:
        run = 0
        for s in st:
            if s == SEED:
                continue
            run += 1
            if s == BOUNDARY:
                hist[run] = hist.get(run, 0) + 1
                run = 0
        if run:  # trailing partial (stop-trimmed) block
            hist[run] = hist.get(run, 0) + 1
    return hist


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
    ap.add_argument("--check-dspark-stream", action="store_true",
                    help="Stage-M2 T1: also assert meta_info['dspark_accept_state'] self-consistency")
    ap.add_argument("--viz", action="store_true",
                    help="visualize dspark_accept_state: per-sample strip + aggregate "
                         "per-slot accept rate + block-length histogram (implies --check-dspark-stream)")
    ap.add_argument("--viz-width", type=int, default=100,
                    help="max glyphs per per-sample strip (default 100)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.viz:
        args.check_dspark_stream = True

    tok = AutoTokenizer.from_pretrained(args.target)
    rows = load_task(args.task, args.n, args.seed)
    print(f"[cfg] task={args.task} n={len(rows)} seed={args.seed} "
          f"temp={args.temperature} enable_thinking={args.enable_thinking}")

    total_completion = 0
    total_verify = 0
    per_sample = []
    stream_checks = {"ok": 0, "fail": 0, "missing": 0}
    all_streams: list[list[int]] = []  # collected for aggregate viz
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
        sample = {"idx": i, "completion_tokens": ct, "spec_verify_ct": vc,
                  "accept_length": al}

        if args.check_dspark_stream:
            st = mi.get("dspark_accept_state")
            if st is None:
                stream_checks["missing"] += 1
                print(f"  [{i}] MISSING dspark_accept_state in meta_info")
            else:
                # Codes: 0=ACCEPT, 1=COMMIT_BOUNDARY, 2=PREFILL_SEED.
                n_boundary = sum(1 for s in st if s == 1)
                n_seed = sum(1 for s in st if s == 2)
                ends_boundary = len(st) > 0 and st[-1] == 1
                # With the prefill seed the stream is per-output-token aligned, so
                # its length equals completion_tokens exactly (crop == output_ids crop).
                len_ok = len(st) == ct
                seed_ok = n_seed == 1 and st[0] == 2
                # spec_verify_ct counts every decode round. A stop/EOS trim can land
                # mid-round, cropping that round's boundary: then #boundary == vc-1
                # AND the stream must end on a non-boundary (partial-block evidence).
                if n_boundary == vc:
                    bnd_ok = True
                elif n_boundary == vc - 1:
                    bnd_ok = not ends_boundary  # trimmed final round -> partial tail
                else:
                    bnd_ok = False
                # Reconstructed decode rounds == vc; accept_len == completion/verify.
                recon_al = ct / vc
                al_ok = abs(recon_al - al) < 1e-6
                ok = len_ok and seed_ok and bnd_ok and al_ok
                stream_checks["ok" if ok else "fail"] += 1
                sample.update({"stream_len": len(st), "stream_boundaries": n_boundary,
                               "stream_seed": n_seed, "stream_ends_boundary": ends_boundary,
                               "stream_ok": ok})
                all_streams.append(st)
                if not ok or i < 5 or args.viz:
                    tail = "bnd" if ends_boundary else "partial"
                    print(f"  [{i}] len={len(st)}(exp {ct}) seed={n_seed} "
                          f"#bnd={n_boundary}(exp {vc} or {vc-1}+{tail}) "
                          f"accept_len={al:.3f} -> {'OK' if ok else 'FAIL'}")
                if args.viz:
                    print(f"      {render_stream_strip(st, args.viz_width)}")

        per_sample.append(sample)
        if (i < 5 or i % 10 == 0) and not args.check_dspark_stream:
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

    stream_ok = True
    if args.check_dspark_stream:
        print(f"\n=== T1 dspark_accept_state self-consistency ===")
        print(f"  ok={stream_checks['ok']} fail={stream_checks['fail']} "
              f"missing={stream_checks['missing']}")
        stream_ok = (stream_checks["fail"] == 0 and stream_checks["missing"] == 0
                     and stream_checks["ok"] > 0)
        print("  RESULT: T1 SERVER STREAM OK" if stream_ok
              else "  RESULT: T1 SERVER STREAM FAILED")

    if args.viz and all_streams:
        # block length = accepts + 1 bonus, so max committed length == gamma+1;
        # slots (in-block draft positions) run 0..gamma-1.
        hist = block_length_histogram(all_streams)
        max_block_len = max(hist) if hist else 1
        gamma = max(1, max_block_len - 1)

        print(f"\n=== dspark_accept_state visualization "
              f"(legend: {_c('36','▶')}seed {_c('32','•')}accept {_c('31','|')}boundary) ===")

        print(f"\n  per-slot accept rate (across {len(all_streams)} samples, "
              f"gamma={gamma}; monotone ↓ is healthy, cf. §4.2 golden):")
        rates = per_slot_accept_rate(all_streams, gamma)
        for i, r in enumerate(rates):
            print(f"    slot {i}: {render_bar(r)} {r:.3f}")

        print(f"\n  block-length histogram (committed tokens per verify round):")
        total_blocks = sum(hist.values())
        max_count = max(hist.values()) if hist else 1
        for L in range(1, max_block_len + 1):
            cnt = hist.get(L, 0)
            bar = "█" * int(round(cnt / max_count * 30))
            pct = 100.0 * cnt / total_blocks if total_blocks else 0.0
            marker = "  <- full block (γ+1)" if L == max_block_len else ""
            print(f"    len {L:2d}: {bar:<30} {cnt:5d} ({pct:4.1f}%){marker}")

    if args.out:
        payload = {
            "config": vars(args),
            "micro_accept_length": micro,
            "macro_accept_length": macro,
            "total_completion": total_completion,
            "total_verify": total_verify,
            "per_sample": per_sample,
        }
        if args.viz and all_streams:
            # persist raw streams for offline re-plotting / regression diffing
            payload["dspark_accept_states"] = all_streams
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"  wrote {args.out}")
    return 0 if stream_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
