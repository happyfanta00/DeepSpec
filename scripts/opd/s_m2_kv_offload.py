#!/usr/bin/env python3
"""Stage-M2 §5.6c smoke — KV-cache offload (tag-selective release/resume) memory verification.

Verifies the "large KV pool for rollout, released to training each step" time-multiplexing:
the tp=8 DSPARK server launches with a LARGE KV pool (--mem-fraction-static 0.6) for fast
rollout, then releases ONLY the KV cache (tags=["kv_cache"]) so training reclaims that budget,
keeping the (cpu-backup'd) weights RESIDENT the whole time. This avoids weight export/import
churn and the weight-corruption class of bugs (only KV pages move; weights never do).

What this asserts (the memory footprint moves as expected, and rollout stays correct):
  1. LAUNCH: big pool up -> rollout-state peak is LARGE (weights + big KV + cuda_graph).
  2. generate() at eval-golden caliber works + accept_state stream self-consistent.
  3. release(tags=["kv_cache"]): per-GPU used memory DROPS substantially (the KV pool is freed;
     weights stay). We assert the drop is a meaningful fraction of the KV pool size.
  4. resume(tags=["kv_cache"]): memory RISES back near the launch footprint (fresh EMPTY pool).
  5. generate() AGAIN after resume works + stream self-consistent (empty pool rebuilt correctly;
     no weight corruption — the whole point of KV-only release vs full release/resume).
  6. Repeat release/resume a few cycles: no leak (used memory returns to the same band each time).

Run (unified env, single node, all 8 GPUs visible):
  CUDA_HOME=/opt/pytorch/cuda CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PYTHONPATH=<DeepSpec>:<DeepSpec>/third_party/verl \
    ~/.venv/dspark-opd-sglang/bin/python scripts/opd/s_m2_kv_offload.py \
      --target Qwen/Qwen3-4B \
      --draft /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest \
      --tp 8 --n 4 --mem-fraction-static 0.6 --cycles 3
# Expected last line: RESULT: KV-OFFLOAD OK
#   released-state used << rollout-state used (KV pool freed, weights resident);
#   resumed-state used ≈ rollout-state used; generate OK before AND after each cycle.

Notes:
  - recipe.dspark_opd MUST be imported first (applies the sglang/transformers compat shims).
  - Reuses s_m2_t3a_server helpers (prompt load, stream check, gpu mem) by import.
"""
from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft",
                    default="/mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest")
    ap.add_argument("--tp", type=int, default=8, help="tensor-parallel size (== #GPUs)")
    ap.add_argument("--n", type=int, default=4, help="#gsm8k prompts to drive per generate phase")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--enable-thinking", action="store_true", default=False)
    # LARGE pool: the KV-offload design point (fast rollout, released to training each step).
    ap.add_argument("--mem-fraction-static", type=float, default=0.6)
    ap.add_argument("--cycles", type=int, default=3, help="#release/resume cycles for leak check")
    # A meaningful KV release should free at least this many MiB/GPU (0.6 pool on H100 is ~tens of
    # GB; we set a conservative floor so a no-op release fails the test).
    ap.add_argument("--min-drop-mib", type=int, default=4096,
                    help="min per-GPU MiB drop after release(kv) to count as a real free")
    ap.add_argument("--port", type=int, default=30000)
    args = ap.parse_args()

    # (0) compat shims FIRST, then ray + reuse the T3a smoke helpers.
    import os

    import recipe.dspark_opd  # noqa: F401
    import ray
    from transformers import AutoTokenizer

    from recipe.dspark_opd.sglang_server import DSparkSglangServer
    # reuse the T3a smoke helpers (same dir)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from s_m2_t3a_server import _check_stream, _gpu_used_mib, _load_prompts

    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not devices:
        devices = ",".join(str(i) for i in range(args.tp))
    dev_list = [d for d in devices.split(",") if d.strip() != ""]
    assert len(dev_list) >= args.tp, (
        f"need >= tp={args.tp} visible GPUs, CUDA_VISIBLE_DEVICES={devices!r}"
    )
    devices = ",".join(dev_list[: args.tp])

    print(f"[cfg] target={args.target} draft={args.draft} tp={args.tp} devices={devices} "
          f"n={args.n} mem_fraction={args.mem_fraction_static} cycles={args.cycles}")

    tok = AutoTokenizer.from_pretrained(args.target)
    prompts = _load_prompts(tok, args.n, args.enable_thinking)
    print(f"[1/6] loaded {len(prompts)} gsm8k prompts (chat template, golden caliber)")

    ray.init(ignore_reinit_error=True, log_to_driver=True)

    print(f"[2/6] launching DSparkSglangServer actor (tp={args.tp}, LARGE pool "
          f"mem_fraction={args.mem_fraction_static}) ...")
    server = DSparkSglangServer.options(
        runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
        name="dspark_sglang_server_kv_offload_smoke",
    ).remote(
        target_path=args.target, draft_path=args.draft, tp_size=args.tp,
        cuda_visible_devices=devices, port=args.port,
        mem_fraction_static=args.mem_fraction_static,
    )
    host, port = ray.get(server.get_address.remote())
    print(f"      server up at http://{host}:{port}")

    sp = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
          "top_p": 1.0, "top_k": -1, "min_p": 0.0}   # §2.4 golden caliber

    def _drive(tag: str) -> bool:
        """Generate n prompts, assert output + stream self-consistency. Returns all-ok."""
        ok_all = True
        tot_ct = tot_vc = 0
        for i, p in enumerate(prompts):
            d = ray.get(server.generate.remote(p, sp))
            mi = d.get("meta_info", {})
            oids = d.get("output_ids")
            ok_shape = isinstance(oids, list) and len(oids) > 0
            ok_stream, detail = _check_stream(mi)
            tot_ct += mi.get("completion_tokens", 0) or 0
            tot_vc += mi.get("spec_verify_ct", 0) or 0
            ok_all = ok_all and ok_shape and ok_stream
            print(f"    [{tag} {i}] out={len(oids) if ok_shape else 'BAD'} {detail} "
                  f"-> {'OK' if (ok_shape and ok_stream) else 'FAIL'}")
        al = (tot_ct / tot_vc) if tot_vc else 0.0
        print(f"    [{tag}] aggregate accept_len={al:.3f} (gsm8k golden ~6.2)")
        return ok_all

    def _used() -> int:
        """Worst-case per-GPU used MiB across the tp devices (nvidia-smi truth)."""
        m = _gpu_used_mib(devices)
        return max(m.values()) if m else 0

    rc = 1
    try:
        # (3) rollout-state footprint (big pool) + generate works.
        time.sleep(1)
        used_launch = _used()
        print(f"[3/6] rollout-state peak (big pool): {used_launch}MiB/GPU")
        ok_gen0 = _drive("launch")
        used_rollout = _used()
        print(f"      after generate: {used_rollout}MiB/GPU")

        # (4) release ONLY the KV cache; weights stay resident. Memory must DROP.
        print("[4/6] release(tags=['kv_cache']) — free KV pool, keep weights resident ...")
        ray.get(server.release.remote(tags=["kv_cache"]))
        time.sleep(2)
        used_released = _used()
        drop = used_rollout - used_released
        drop_ok = drop >= args.min_drop_mib
        print(f"      released-state used: {used_released}MiB/GPU "
              f"(drop {drop}MiB vs rollout; need >= {args.min_drop_mib}MiB) "
              f"-> {'OK (KV freed)' if drop_ok else 'BAD (no meaningful free)'}")

        # (5) resume KV (fresh empty pool) + generate again — proves no weight corruption.
        print("[5/6] resume(tags=['kv_cache']) — rebuild empty KV pool + re-generate ...")
        ray.get(server.resume.remote(tags=["kv_cache"]))
        time.sleep(1)
        used_resumed = _used()
        rise_ok = used_resumed >= used_released + args.min_drop_mib // 2
        print(f"      resumed-state used: {used_resumed}MiB/GPU "
              f"(vs released {used_released}MiB) -> {'OK (pool back)' if rise_ok else 'BAD (no rebuild)'}")
        ok_gen1 = _drive("resumed")

        # (6) leak check: cycle release/resume a few times, used memory returns to the same band.
        print(f"[6/6] leak check: {args.cycles} release/resume cycles ...")
        bands = [used_resumed]
        cyc_gen_ok = True
        for c in range(args.cycles):
            ray.get(server.release.remote(tags=["kv_cache"]))
            time.sleep(1)
            u_rel = _used()
            ray.get(server.resume.remote(tags=["kv_cache"]))
            time.sleep(1)
            u_res = _used()
            bands.append(u_res)
            # a single-prompt generate each cycle to keep proving correctness under churn
            d = ray.get(server.generate.remote(prompts[0], sp))
            ok_c, _ = _check_stream(d.get("meta_info", {}))
            cyc_gen_ok = cyc_gen_ok and ok_c
            print(f"      cycle {c}: released={u_rel}MiB resumed={u_res}MiB "
                  f"gen={'OK' if ok_c else 'FAIL'}")
        band_spread = max(bands) - min(bands)
        leak_ok = band_spread <= args.min_drop_mib // 2   # resumed footprint stable (no creep)
        print(f"      resumed-band spread {band_spread}MiB over {len(bands)} points "
              f"-> {'OK (no leak)' if leak_ok else 'BAD (memory creep)'}")

        all_ok = (ok_gen0 and ok_gen1 and drop_ok and rise_ok and cyc_gen_ok and leak_ok)
        print(f"\n[verdict] gen(launch)={ok_gen0} gen(resumed)={ok_gen1} "
              f"drop_ok={drop_ok} rise_ok={rise_ok} cycle_gen_ok={cyc_gen_ok} leak_ok={leak_ok}")
        rc = 0 if all_ok else 1
    finally:
        print("[cleanup] shutting down server actor ...")
        try:
            ray.get(server.shutdown.remote())
        except Exception as e:  # noqa: BLE001
            print(f"  shutdown error (ignored): {e!r}")
        ray.kill(server)

    print("RESULT: KV-OFFLOAD OK" if rc == 0 else "RESULT: KV-OFFLOAD FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
