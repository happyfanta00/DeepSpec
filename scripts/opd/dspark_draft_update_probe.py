#!/usr/bin/env python3
"""Stage-M2 T2 probe: DSPARK draft-only weight update (draft changes, target frozen).

Verifies the upstream sglang change that lets DSpark-OPD RL push freshly-trained
draft weights back into the running spec-decode engine every step WITHOUT touching
the frozen target (design §5.2 / §5.5). Before this change, DSparkWorkerV2 had no
``update_weights_from_tensor`` so the request fell through ``__getattr__`` to the
target worker: the draft never updated. We added:
  - io_struct ``UpdateWeightsFromTensorReqInput.draft_model_only``
  - ``DSparkWorkerV2.update_weights_from_tensor`` (routes tensors to the draft
    model runner; skips the target when draft_model_only=True)
  - the flag threaded through Engine + HttpServerEngineAdapter

This probe drives the server through ``HttpServerEngineAdapter`` -- the same class
verl's async rollout uses -- so it exercises the exact CUDA-IPC weight-sync path
T6 will use. It reads the draft ckpt tensors from disk (names already validated
47/47 in §4.0), perturbs a few, pushes them draft-only, and proves the effect via
``/weights_checker`` checksums (which cover both ``target`` and ``draft.``-prefixed
params).

Assertions:
  (a) draft weights CHANGED  -> some ``draft.*`` checksum differs after the push
  (b) target weights FROZEN  -> every non-``draft.`` checksum identical
  (c) restore round-trips    -> pushing the originals back returns to the initial
      checksum (draft-only update is exact, no target contamination)

Two modes:
  - single-ckpt (default): perturb a few draft tensors *1.05, push draft-only,
    verify draft.* checksum changed / target frozen / restore round-trips.
  - behavioral two-ckpt (--draft-b): server starts on A, pushes B draft-only over
    the exact T6/verl update path, and re-checksums to prove the weights swapped
    (draft changed, target frozen). Optionally also visualizes q(A) vs q(B) on the
    first verify block IF the temporary q-probe is present (see below); the probe
    was removed after T2 verification passed, so by default (c) is SKIPPED and the
    verdict rests on the checksum-based (a)+(b).

  === how to re-visualize q(A) vs q(B) (re-add the temporary probe) ===
  The probe was removed from upstream sglang after T2 passed. To reproduce the
  q-divergence table, temporarily re-add it to
    third_party/sglang/python/sglang/srt/speculative/dspark_components/dspark_verify.py
  In accept_draft_tokens(), right after `draft_probs = SoftmaxTemp.execute(...)`,
  gather q for the proposed tokens and append one JSON line per verify call to
  SGLANG_DSPARK_PROBE_FILE (env), resetting a per-round counter when that file is
  truncated. Concretely, drop in a helper like:
      import json, os
      _LIM = int(os.environ.get("SGLANG_DSPARK_PROBE_Q", "0"))
      _F = os.environ.get("SGLANG_DSPARK_PROBE_FILE", "")
      _calls = {"n": 0, "sz": 0}
      def _probe(draft_probs, draft_tokens):   # draft_tokens[b,i] ~ corrected_logits[b,i]
          if _LIM <= 0: return
          sz = os.path.getsize(_F) if _F and os.path.exists(_F) else 0
          if sz < _calls["sz"]: _calls["n"] = 0          # re-arm on truncate
          _calls["sz"] = sz
          if _calls["n"] >= _LIM: return
          _calls["n"] += 1
          bs, g, _ = draft_probs.shape
          prop = draft_tokens.view(bs, -1)[:, :g]
          q = draft_probs.gather(-1, prop.unsqueeze(-1).long())[..., 0]
          for b in range(bs):
              rec = {"proposed_tokens":[int(x) for x in prop[b].tolist()],
                     "q_on_proposed":[round(float(x),6) for x in q[b].tolist()],
                     "top1_prob":[round(float(x),6) for x in draft_probs[b].max(-1).values.tolist()]}
              print("[DSPARK_PROBE_Q]", rec, flush=True)
              if _F:
                  open(_F,"a").write(json.dumps(rec)+"\n")
  and call `_probe(draft_probs, draft_block.draft_tokens)` on the next line. This
  script arms it automatically (sets SGLANG_DSPARK_PROBE_Q/_FILE) in --draft-b mode;
  once the probe is back, (c) shows the per-slot q(A) vs q(B) table again.
  IMPORTANT: gather q at draft_block.draft_tokens (aligned with corrected_logits),
  NOT at `candidates` (strided verify buffer; anchor at slot 0 -> off-by-one, q==0).

Runs on 1 GPU, no verl. The adapter launches the server subprocess itself, so do
NOT pre-launch launch_sglang_upstream_dspark.sh; just run this. Use the sglang env.

Usage:
    # single-ckpt update-path check
    CUDA_HOME=/opt/pytorch/cuda CUDA_VISIBLE_DEVICES=0 \
      ~/.venv/dspark-opd-sglang/bin/python scripts/opd/dspark_draft_update_probe.py \
        --target Qwen/Qwen3-4B \
        --draft /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest
    # -> RESULT: T2 DRAFT-ONLY UPDATE OK

    # behavioral two-ckpt (checksum verdict; q table only if probe re-added)
    CUDA_HOME=/opt/pytorch/cuda CUDA_VISIBLE_DEVICES=0 \
      ~/.venv/dspark-opd-sglang/bin/python scripts/opd/dspark_draft_update_probe.py \
        --target Qwen/Qwen3-4B \
        --draft /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest \
        --draft-b /mnt/scratch/checkpoints/deepspec/dspark_opd_qwen3_4b/step_16000
    # -> RESULT: T2 TWO-CKPT UPDATE OK
"""
from __future__ import annotations

import argparse
import os


def _checksums_by_prefix(payload):
    """Split a /weights_checker checksum payload into (target, draft) name->chk dicts.

    check_weights merges draft checksums under a ``draft.`` prefix
    (weight_updater.py:_merge_checksum_payloads). Under TP>1 the payload is a list
    of per-rank dicts; we launch TP=1 here so handle the single dict (or list[0]).
    """
    if isinstance(payload, list):
        payload = payload[0]
    checksums = payload["checksums"]
    target, draft = {}, {}
    for name, chk in checksums.items():
        (draft if name.startswith("draft.") else target)[name] = chk
    return target, draft


def _load_draft_named_tensors(ckpt_dir, torch, load_file):
    """All draft weights (bf16, cuda) named as Qwen3DSparkModel.load_weights expects,
    dropping embed/lm_head/rotary which load_weights skips (§4.0)."""
    import os as _os
    raw = load_file(_os.path.join(ckpt_dir, "model.safetensors"))
    out = []
    for k, v in raw.items():
        if k.startswith(("embed_tokens.", "lm_head.", "rotary_emb.")):
            continue
        out.append((k, v.to(torch.bfloat16).cuda().contiguous()))
    return out


def _read_probe_records(path, since_line, want, timeout_s=30.0):
    """Read JSON q-records the server appended after line ``since_line``, waiting
    until ``want`` new ones exist (the server writes them from another process).

    Returns (new_records, new_cursor) where new_cursor is the total line count read.
    """
    import json
    import time
    deadline = time.perf_counter() + timeout_s
    while True:
        lines = []
        if os.path.exists(path):
            with open(path) as f:
                lines = f.read().splitlines()
        new = [json.loads(ln) for ln in lines[since_line:] if ln.strip()]
        if len(new) >= want or time.perf_counter() >= deadline:
            return new, len(lines)
        time.sleep(0.3)


def _fmt_bar(x, width=20):
    n = int(round(max(0.0, min(1.0, x)) * width))
    return "█" * n + "░" * (width - n)


def _visualize_q_diff(rec_a, rec_b):
    """Side-by-side per-slot q(A) vs q(B) for the first-block verify record."""
    toks_a = rec_a["proposed_tokens"]
    qa = rec_a["q_on_proposed"]
    qb = rec_b["q_on_proposed"]
    toks_b = rec_b["proposed_tokens"]
    t1a, t1b = rec_a["top1_prob"], rec_b["top1_prob"]
    n = min(len(qa), len(qb))
    print("\n  ── first-block draft q: A(base) vs B(opd), same anchor & context ──")
    print(f"  {'slot':>4} {'tokA':>6} {'q(A)':>8} {'q(B)':>8} {'Δq':>9}   "
          f"{'top1(A)':>8} {'top1(B)':>8}  q(A)->q(B)")
    sum_abs_dq = 0.0
    tok_diff = 0
    for i in range(n):
        dq = qb[i] - qa[i]
        sum_abs_dq += abs(dq)
        same_tok = (i < len(toks_b) and toks_a[i] == toks_b[i])
        tok_diff += 0 if same_tok else 1
        arrow = f"{_fmt_bar(qa[i])}|{_fmt_bar(qb[i])}"
        print(f"  {i:>4} {toks_a[i]:>6} {qa[i]:>8.4f} {qb[i]:>8.4f} {dq:>+9.4f}   "
              f"{t1a[i]:>8.4f} {t1b[i]:>8.4f}  {arrow}")
    mean_abs_dq = sum_abs_dq / n if n else 0.0
    print(f"\n  mean |Δq on proposed token| = {mean_abs_dq:.4f}   "
          f"(proposed-token argmax differs on {tok_diff}/{n} slots)")
    return mean_abs_dq


def _run_behavioral(engine, args, check, t0, d0, torch, load_file) -> int:
    """Server started on A; push B draft-only; drive the same prompt before/after,
    then read the captured q records and visualize q(A) vs q(B)."""
    from transformers import AutoTokenizer

    probe_file = args._probe_file
    tok = AutoTokenizer.from_pretrained(args.target)
    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": args.probe_prompt}],
        add_generation_prompt=True, enable_thinking=False, tokenize=True,
    )
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    input_ids = [int(t) for t in input_ids]
    sp = {"temperature": 1.0, "max_new_tokens": 64,
          "top_p": 1.0, "top_k": -1, "min_p": 0.0}

    def drive_and_read(tag):
        # Truncate the probe file first: this is the RE-ARM signal. The server-side
        # counter is per-process and only fires for the first N verify calls; it
        # resets when it sees the file shrink, so each drive captures fresh q.
        open(probe_file, "w").close()
        print(f"\n[drive:{tag}] generating + capturing draft q (first block) ...")
        engine.generate(input_ids=input_ids, sampling_params=sp)
        recs, _ = _read_probe_records(probe_file, 0, want=1)
        if not recs:
            print(f"      WARN: no q record captured for {tag}")
            return None
        return recs[0]  # first-block verify record

    print("[3/5] driving prompt with draft A (capturing q(A)) ...")
    rec_a = drive_and_read("A")

    print(f"[4/5] pushing draft B ({args.draft_b}) draft_model_only=True + flush cache ...")
    named_b = _load_draft_named_tensors(args.draft_b, torch, load_file)
    # flush_cache=True clears the radix/KV cache as part of the update, so drive:B
    # RECOMPUTES the first block (otherwise it reuses drive:A's cached prompt+block
    # result and the verify forward -- hence q(B) -- never re-runs). Do NOT call
    # engine.flush_cache() separately: /flush_cache returns plain text, not JSON,
    # and the adapter's .json() would crash on it.
    ok = engine.update_weights_from_tensor(named_b, draft_model_only=True, flush_cache=True)
    print(f"      update returned: {ok}")
    t1, d1 = check()
    draft_changed = [k for k in d0 if d0[k] != d1.get(k)]
    target_changed = [k for k in t0 if t0[k] != t1.get(k)]

    print("[5/5] driving SAME prompt with draft B (capturing q(B)) ...")
    rec_b = drive_and_read("B")

    a_ok = len(draft_changed) > 0
    b_ok = len(target_changed) == 0

    # q(A) vs q(B) visualization is OPTIONAL: it needs the temporary verify-kernel
    # q-probe, which was removed after T2 verification passed (see this file's
    # header for how to re-add it). Without the probe, drive_and_read returns None
    # and we simply skip (c); the permanent verdict is the checksum-based (a)+(b).
    have_q = bool(rec_a and rec_b)
    q_ok = True
    print()
    print(f"  (a) draft CHANGED after B push       : {'OK' if a_ok else 'FAIL'} "
          f"({len(draft_changed)} draft params differ from A)")
    print(f"  (b) target FROZEN on draft-only push : {'OK' if b_ok else 'FAIL'} "
          f"({len(target_changed)} target params changed)")
    if have_q:
        mean_abs_dq = _visualize_q_diff(rec_a, rec_b)
        q_ok = mean_abs_dq > 1e-4
        print(f"  (c) draft q CHANGED in-engine        : {'OK' if q_ok else 'FAIL'} "
              f"(mean |Δq| = {mean_abs_dq:.4f} on first block)")
    else:
        print("  (c) draft q divergence               : SKIPPED (verify-kernel "
              "q-probe not present; re-add it per this script's header to see q(A) vs q(B))")

    ok = a_ok and b_ok and q_ok
    if ok:
        tail = ", draft predictions changed in-engine)" if have_q else ")"
        print("RESULT: T2 TWO-CKPT UPDATE OK (weights swapped A->B, target frozen" + tail)
    else:
        print("RESULT: T2 TWO-CKPT UPDATE FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft",
                    default="/mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest")
    ap.add_argument("--draft-b", default=None,
                    help="second ckpt (e.g. opd/step_16000). If set, runs the BEHAVIORAL "
                         "two-ckpt mode: server starts on --draft (A), pushes --draft-b (B) "
                         "draft-only, and re-checksums to prove draft changed / target frozen. "
                         "The q(A) vs q(B) table is shown only if the temporary verify q-probe "
                         "is re-added (see this script's header); otherwise (c) is SKIPPED.")
    ap.add_argument("--probe-prompt", default="What is 17 times 24? Show your steps.",
                    help="prompt driven before/after the swap to fire the verify q-probe")
    ap.add_argument("--probe-q-calls", type=int, default=1,
                    help="how many verify calls to capture draft q for (per generation)")
    ap.add_argument("--probe-file", default=None,
                    help="where the server writes q records (default: a temp file)")
    ap.add_argument("--attention-backend", default="flashinfer")
    ap.add_argument("--mem-fraction-static", type=float, default=0.6)
    ap.add_argument("--port", type=int, default=30010)
    ap.add_argument("--n-perturb", type=int, default=3,
                    help="how many draft tensors to perturb (single-ckpt mode only)")
    args = ap.parse_args()

    import tempfile

    import torch
    from safetensors.torch import load_file
    from sglang.srt.entrypoints.http_server_engine import HttpServerEngineAdapter

    # Behavioral mode arms the server-side verify q-probe BEFORE the subprocess
    # launches. The probe both prints and appends JSON lines to PROBE_FILE, which
    # this driver reads back to visualize q(A) vs q(B) itself -- no log scraping.
    if args.draft_b:
        os.environ["SGLANG_DSPARK_PROBE_Q"] = str(args.probe_q_calls)
        probe_file = args.probe_file or os.path.join(
            tempfile.gettempdir(), f"dspark_probe_q_{args.port}.jsonl"
        )
        # start clean so we only read this run's records
        try:
            os.remove(probe_file)
        except FileNotFoundError:
            pass
        os.environ["SGLANG_DSPARK_PROBE_FILE"] = probe_file
        args._probe_file = probe_file

    print(f"[cfg] target={args.target} draft(A)={args.draft} attn={args.attention_backend}")
    if args.draft_b:
        print(f"[cfg] draft(B)={args.draft_b}  (behavioral two-ckpt mode; "
              f"q captured to {os.environ['SGLANG_DSPARK_PROBE_FILE']})")
    print("[1/6] launching DSPARK server via HttpServerEngineAdapter ...")
    engine = HttpServerEngineAdapter(
        model_path=args.target,
        speculative_algorithm="DSPARK",
        speculative_draft_model_path=args.draft,
        speculative_draft_attention_backend=args.attention_backend,
        attention_backend=args.attention_backend,
        mem_fraction_static=args.mem_fraction_static,
        trust_remote_code=True,
        disable_cuda_graph=True,  # probe only updates weights; skip graph capture
        port=args.port,
    )

    rc = 1
    try:
        def check():
            # endpoint is /weights_checker; body -> {success, message, ranks:[...],
            # per_engine_checksum}. Each rank dict carries per-param `checksums`
            # (draft params prefixed `draft.`).
            r = engine._make_request("weights_checker", {"action": "checksum"})
            assert r.get("success"), f"weights_checker failed: {r.get('message')}"
            return _checksums_by_prefix(r["ranks"])

        print("[2/6] snapshotting initial checksums (/weights_checker) ...")
        t0, d0 = check()
        print(f"      target params={len(t0)}  draft params={len(d0)}")
        if not d0:
            print("FATAL: no draft.* checksums returned; is this a DSPARK engine?")
            return 2

        # ---- BEHAVIORAL two-ckpt mode: server starts on A, pushes B draft-only,
        # drives the same prompt before/after so the verify q-probe shows q(A) vs
        # q(B) for the first block. This exercises the *exact* T6/verl update path
        # (HttpServerEngineAdapter + draft_model_only) inside the engine. ----
        if args.draft_b:
            rc = _run_behavioral(engine, args, check, t0, d0, torch, load_file)
            return rc

        print(f"[3/6] loading draft ckpt tensors and perturbing {args.n_perturb} ...")
        raw = load_file(os.path.join(args.draft, "model.safetensors"))
        # skip embed/lm_head/rotary -- Qwen3DSparkModel.load_weights drops them (§4.0)
        pick = [k for k in raw.keys()
                if not k.startswith(("embed_tokens.", "lm_head.", "rotary_emb."))][: args.n_perturb]
        originals, perturbed = [], []
        for k in pick:
            t = raw[k].to(torch.bfloat16).cuda()
            originals.append((k, t.clone()))
            perturbed.append((k, (t * 1.05).contiguous()))
        print(f"      perturbing: {pick}")

        print("[4/6] update_weights_from_tensor(draft_model_only=True) ...")
        ok = engine.update_weights_from_tensor(perturbed, draft_model_only=True)
        print(f"      update returned: {ok}")
        t1, d1 = check()

        draft_changed = [k for k in d0 if d0[k] != d1.get(k)]
        target_changed = [k for k in t0 if t0[k] != t1.get(k)]
        print(f"[5/6] draft changed={len(draft_changed)} (expect>0)  "
              f"target changed={len(target_changed)} (expect 0)")

        print("[6/6] restoring original draft weights (draft_model_only=True) ...")
        engine.update_weights_from_tensor(originals, draft_model_only=True)
        t2, d2 = check()
        restore_ok = all(d0[k] == d2.get(k) for k in d0) and all(t0[k] == t2.get(k) for k in t0)

        a_ok = len(draft_changed) > 0
        b_ok = len(target_changed) == 0
        print()
        print(f"  (a) draft CHANGED on push          : {'OK' if a_ok else 'FAIL'} "
              f"({len(draft_changed)} params)")
        print(f"  (b) target FROZEN on draft-only push: {'OK' if b_ok else 'FAIL'} "
              f"({len(target_changed)} target params changed)")
        print(f"  (c) restore round-trips exactly     : {'OK' if restore_ok else 'FAIL'}")

        if a_ok and b_ok and restore_ok:
            print("RESULT: T2 DRAFT-ONLY UPDATE OK")
            rc = 0
        else:
            print("RESULT: T2 DRAFT-ONLY UPDATE FAILED")
    finally:
        engine.shutdown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
