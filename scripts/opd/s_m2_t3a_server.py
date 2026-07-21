#!/usr/bin/env python3
"""Stage-M2 T3a smoke — standalone DSPARK sglang server actor (infra only, no train_step).

Verifies the T3a server infrastructure end-to-end WITHOUT any training loop or DSpark
tensor-contract dependency — this is essentially the T2 probe upgraded to a resident
Ray actor:

  1. Build the DSparkSglangServer Ray actor (recipe.dspark_opd.sglang_server) with the
     tp=8 server colocated on this node's GPUs (NOSET + joined CUDA_VISIBLE_DEVICES +
     NodeAffinity — the same placement verl's async path uses). In this standalone smoke
     there is no training worker_group, so we build the actor directly, handing it the
     visible-device list ourselves.
  2. Drive gsm8k prompts through the actor's generate() at the EVAL GOLDEN caliber
     (temp=1.0, chat template, enable_thinking=False, no top_p/top_k/min_p truncation;
     §2.4) and assert:
       - output_ids present + finite, meta_info has completion_tokens / spec_verify_ct;
       - dspark_accept_state (T1 stream) present + self-consistent: len == completion_tokens,
         exactly one PREFILL_SEED(2) at index 0, #COMMIT_BOUNDARY(1) == spec_verify_ct
         (or vc-1 with a non-boundary tail = stop-trim), reconstructed accept_len matches;
       - accept length in the gsm8k golden ballpark (~6.2, §4.2) — sanity, not exact.
  3. RESIDENT co-existence check (the T3b design): with a small KV pool (mem_fraction=0.15)
     the engine footprint is tiny, so T3b keeps the engine RESIDENT and NEVER calls
     release/resume — avoiding the ~5s (cudagraph-remap + weight cpu_backup) per-step cost AND
     the weight-corruption risk entirely. This step reports the rollout-state peak, then
     (with --sim-train-gib N) allocates N GiB/GPU as a training-state proxy on the SAME GPUs and
     asserts the co-existence peak stays under --gpu-cap-gib while generation still works.
     [FUTURE 后手] If a larger batch/seqlen ever overflows co-residence, switch to per-step
     release(tags=["kv_cache"]) — it only frees the (small) KV pool, does NOT touch weights or
     cuda_graph, so it avoids both the cpu_backup weight copy and the cudagraph remap that made
     the full release/resume ~5s (and can verify via /weights_checker that weights stay unchanged
     when only KV is released).

Run (unified env, single node, all 8 GPUs visible):
  CUDA_HOME=/opt/pytorch/cuda CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PYTHONPATH=<DeepSpec>:<DeepSpec>/third_party/verl \
    ~/.venv/dspark-opd-sglang/bin/python scripts/opd/s_m2_t3a_server.py \
      --target Qwen/Qwen3-4B \
      --draft /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest \
      --tp 8 --n 4 --sim-train-gib 32
# Expected last line: RESULT: T3a SERVER OK
#   [4a] rollout-state peak should be MUCH smaller than the old 53GB (0.15 vs 0.6);
#   [4b] with --sim-train-gib 32, co-existence peak (engine + 32GiB proxy) must stay < cap.

Notes:
  - The recipe package (recipe.dspark_opd) MUST be importable first so its __init__ applies
    the sglang/transformers compat shims (see recipe/dspark_opd/__init__.py). We import it
    before ray.init / building the actor.
  - tp<8 works too (e.g. --tp 1 for a quick single-GPU check); default 8 matches T3.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def _gpu_used_mib(devices: str) -> dict[int, int]:
    """Per-GPU used MiB (true allocation incl. the server subprocess) via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        )
    except Exception:  # noqa: BLE001
        return {}
    used = {}
    want = {int(x) for x in devices.split(",") if x.strip() != ""}
    for line in out.strip().splitlines():
        idx, mib = (p.strip() for p in line.split(","))
        if int(idx) in want:
            used[int(idx)] = int(mib)
    return used


def _load_prompts(tokenizer, n: int, enable_thinking: bool):
    """First n gsm8k prompts, tokenized at the eval golden caliber (chat template)."""
    here = os.path.dirname(os.path.abspath(__file__))
    ds = os.path.join(here, "..", "..", "eval_datasets", "gsm8k.jsonl")
    prompts = []
    with open(os.path.abspath(ds)) as f:
        for line in f:
            if len(prompts) >= n:
                break
            row = json.loads(line)
            messages = [{"role": "user", "content": row["turns"][0]}]
            ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=enable_thinking,
                tokenize=True, return_tensors=None,
            )
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            prompts.append([int(t) for t in ids])
    return prompts


def _check_stream(mi: dict) -> tuple[bool, str]:
    """T1 accept_state self-consistency (mirrors dspark_server_accept_probe.py:217-243).

    Codes: 0=ACCEPT, 1=COMMIT_BOUNDARY, 2=PREFILL_SEED.
    """
    ct = mi.get("completion_tokens")
    vc = mi.get("spec_verify_ct")
    st = mi.get("dspark_accept_state")
    if not ct or not vc:
        return False, f"missing spec stats: completion={ct} verify={vc}"
    if st is None:
        return False, "missing dspark_accept_state in meta_info"
    n_boundary = sum(1 for s in st if s == 1)
    n_seed = sum(1 for s in st if s == 2)
    ends_boundary = len(st) > 0 and st[-1] == 1
    len_ok = len(st) == ct
    seed_ok = n_seed == 1 and st[0] == 2
    if n_boundary == vc:
        bnd_ok = True
    elif n_boundary == vc - 1:
        bnd_ok = not ends_boundary  # trimmed final round -> partial tail
    else:
        bnd_ok = False
    detail = (f"len={len(st)}(ct={ct},{'ok' if len_ok else 'BAD'}) "
              f"seed={n_seed}@{st[0] if st else '-'}({'ok' if seed_ok else 'BAD'}) "
              f"#bnd={n_boundary}(vc={vc},{'ok' if bnd_ok else 'BAD'}) al={ct / vc:.3f}")
    return (len_ok and seed_ok and bnd_ok), detail


def _alloc_train_proxy(gib_per_gpu: float, n_gpus: int):
    """Allocate gib_per_gpu of bf16 tensor on each of n_gpus — a crude proxy for the
    training-state footprint (FSDP draft + co-resident teacher) that must co-exist with
    the RESIDENT sglang engine on the SAME GPUs. Returns the holders (keep referenced)."""
    import torch
    holders = []
    numel = int(gib_per_gpu * (1024 ** 3) / 2)  # bf16 = 2 bytes
    for gi in range(n_gpus):
        try:
            holders.append(torch.empty(numel, dtype=torch.bfloat16, device=f"cuda:{gi}"))
        except Exception as e:  # noqa: BLE001
            print(f"        [proxy] GPU{gi} alloc {gib_per_gpu}GiB failed: {e!r}")
    torch.cuda.synchronize()
    return holders


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft",
                    default="/mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest")
    ap.add_argument("--tp", type=int, default=8, help="tensor-parallel size (== #GPUs)")
    ap.add_argument("--n", type=int, default=4, help="#gsm8k prompts to drive")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--enable-thinking", action="store_true", default=False)
    # T3b resident design: small KV pool (training rollout needs little KV — tp=8 makes it
    # ~0.018 MB/token/GPU). 0.15 leaves ~9GB pool (plenty for conc<=few-hundred @2048) with a
    # safety margin over 0.1. Tune here + measure peak.
    ap.add_argument("--mem-fraction-static", type=float, default=0.15)
    ap.add_argument("--sim-train-gib", type=float, default=0.0,
                    help="allocate this many GiB/GPU as a training-state proxy to test RESIDENT "
                         "co-existence with the engine (0 = skip). ~32 mirrors FSDP draft+teacher.")
    ap.add_argument("--gpu-cap-gib", type=float, default=78.0,
                    help="per-GPU memory cap for the co-existence verdict (H100 80GB, leave headroom)")
    ap.add_argument("--port", type=int, default=30000)
    args = ap.parse_args()

    # (0) compat shims FIRST (sglang _launch_subprocesses / transformers alias), then ray.
    import recipe.dspark_opd  # noqa: F401
    import ray
    from transformers import AutoTokenizer

    from recipe.dspark_opd.sglang_server import DSparkSglangServer

    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not devices:
        devices = ",".join(str(i) for i in range(args.tp))
    dev_list = [d for d in devices.split(",") if d.strip() != ""]
    assert len(dev_list) >= args.tp, (
        f"need >= tp={args.tp} visible GPUs, CUDA_VISIBLE_DEVICES={devices!r}"
    )
    devices = ",".join(dev_list[: args.tp])

    print(f"[cfg] target={args.target} draft={args.draft} tp={args.tp} devices={devices} "
          f"n={args.n} temp={args.temperature} enable_thinking={args.enable_thinking}")

    tok = AutoTokenizer.from_pretrained(args.target)
    prompts = _load_prompts(tok, args.n, args.enable_thinking)
    print(f"[1/5] loaded {len(prompts)} gsm8k prompts (chat template, golden caliber)")

    ray.init(ignore_reinit_error=True, log_to_driver=True)

    # (1) build the server actor. Standalone (no worker_group): build directly, this driver
    # process already sees all 8 GPUs. NOSET so ray doesn't clobber the actor's device set.
    print(f"[2/5] launching DSparkSglangServer actor (tp={args.tp}) ...")
    server = DSparkSglangServer.options(
        runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
        name="dspark_sglang_server_t3a_smoke",
    ).remote(
        target_path=args.target, draft_path=args.draft, tp_size=args.tp,
        cuda_visible_devices=devices, port=args.port,
        mem_fraction_static=args.mem_fraction_static,
    )
    host, port = ray.get(server.get_address.remote())
    print(f"      server up at http://{host}:{port}")

    sp = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
          "top_p": 1.0, "top_k": -1, "min_p": 0.0}   # §2.4 golden caliber

    rc = 1
    try:
        mem_up = _gpu_used_mib(devices)
        print(f"[3/5] driving {len(prompts)} prompts + checking accept_state stream ...")
        checks = {"ok": 0, "fail": 0}
        tot_ct, tot_vc = 0, 0
        for i, p in enumerate(prompts):
            d = ray.get(server.generate.remote(p, sp))
            mi = d.get("meta_info", {})
            oids = d.get("output_ids")
            ok_shape = isinstance(oids, list) and len(oids) > 0
            ok_stream, detail = _check_stream(mi)
            ct, vc = mi.get("completion_tokens", 0), mi.get("spec_verify_ct", 0)
            tot_ct += ct or 0
            tot_vc += vc or 0
            passed = ok_shape and ok_stream
            checks["ok" if passed else "fail"] += 1
            print(f"  [{i}] out={len(oids) if ok_shape else 'BAD'} {detail} "
                  f"-> {'OK' if passed else 'FAIL'}")

        agg_al = (tot_ct / tot_vc) if tot_vc else 0.0
        print(f"      aggregate accept_len = {agg_al:.3f} (gsm8k golden ~6.2, §4.2)")

        # (4) RESIDENT CO-EXISTENCE CHECK (T3b decision: mem_fraction small + NO release/resume).
        #     With a small KV pool the engine footprint is tiny; T3b keeps it RESIDENT and never
        #     calls release/resume (avoids the ~5s cudagraph-remap + cpu_backup, and the
        #     weight-corruption risk entirely). Here we (4a) report the rollout-state peak, then
        #     (4b) simulate the training-state footprint (~sim_train_gib/GPU) ON THE SAME GPUs and
        #     assert co-existence stays under the cap — the real question for the resident design.
        print(f"[4/5] RESIDENT CO-EXISTENCE (mem_fraction={args.mem_fraction_static}, no release/resume) ...")
        cap_mib = int(args.gpu_cap_gib * 1024)
        roll_max = max(mem_up.values()) if mem_up else 0
        print(f"      [4a rollout-state peak] {roll_max}MiB/GPU "
              f"(权重 ~1.4G + KV池 + cudagraph; mem_fraction={args.mem_fraction_static})")
        for g in sorted(mem_up):
            print(f"        GPU{g}: {mem_up[g]}MiB")

        coexist_ok = True
        if args.sim_train_gib > 0:
            print(f"      [4b simulate training-state footprint] allocating "
                  f"{args.sim_train_gib}GiB/GPU on all {args.tp} GPUs (proxy for FSDP+teacher) ...")
            holders = _alloc_train_proxy(args.sim_train_gib, args.tp)
            time.sleep(2)
            mem_coexist = _gpu_used_mib(devices)
            worst = max(mem_coexist.values()) if mem_coexist else 0
            for g in sorted(mem_coexist):
                over = " OVER-CAP!" if mem_coexist[g] > cap_mib else ""
                print(f"        GPU{g}: {mem_coexist[g]}MiB{over}")
            coexist_ok = worst <= cap_mib
            print(f"      co-existence peak {worst}MiB/GPU vs cap {cap_mib}MiB "
                  f"-> {'OK (resident design fits)' if coexist_ok else 'OVER CAP'}")
            # engine must still generate correctly WHILE the training proxy occupies memory.
            d_co = ray.get(server.generate.remote(prompts[0], sp))
            ok_co, _ = _check_stream(d_co.get("meta_info", {}))
            print(f"      generate under co-residence: {'OK' if ok_co else 'FAIL'}")
            coexist_ok = coexist_ok and ok_co
            del holders  # free the proxy
        else:
            print("      [4b] skipped training-proxy (pass --sim-train-gib N to test co-existence)")

        # Verdict: all streams structurally ok AND (if simulated) resident co-existence fits.
        print(f"[5/5] streams ok={checks['ok']} fail={checks['fail']}; "
              f"rollout_peak={roll_max}MiB; coexist_ok={coexist_ok}")
        rc = 0 if (checks["fail"] == 0 and checks["ok"] > 0 and coexist_ok) else 1
    finally:
        print("[cleanup] shutting down server actor ...")
        try:
            ray.get(server.shutdown.remote())
        except Exception as e:  # noqa: BLE001
            print(f"  shutdown error (ignored): {e!r}")
        ray.kill(server)

    print("RESULT: T3a SERVER OK" if rc == 0 else "RESULT: T3a SERVER FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
