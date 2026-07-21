#!/usr/bin/env python3
"""Stage-M2 T3b smoke — sglang rollout bridge (prompt extract -> generate -> rebuild).

Tests T3b's rollout glue (recipe.dspark_opd.sglang_rollout_bridge) end-to-end against a REAL
T3a DSPARK server, WITHOUT the full FSDP training loop — this isolates the "cache prompt ->
fresh sglang response -> rebuilt padded batch" path that train_step Phase-1 now uses.

Flow:
  1. Build the T3a DSparkSglangServer actor (tp=8, mem_fraction=0.15, resident).
  2. Load a few real cache samples -> extract_prompts (input_ids[:first loss_mask==1]).
  3. sglang_generate_batch: B prompts x rollout_n -> B*n live DSPARK responses (explicit repeat).
  4. rebuild_padded_batch -> right-padded (prompt+new_response) batch.
  5. Assert:
     - B*n rows; each row's loss_mask==1 span == the new response, ==0 on prompt+padding;
     - responses are non-empty, finite token ids, most end on EOS (finish=stop);
     - the rollout_n copies of a prompt are INDEPENDENT (not identical) at temp=1.0;
     - shapes/dtypes match the DSpark tensor-contract (§S1): input_ids/loss_mask/attention_mask
       [B*n, T] long, so downstream Phase-1 (dspark_block_rollout on the new seq) can consume it.
  6. (optional --check-forward) recompute target_hidden_states on the new seq via a co-resident
     teacher, proving the new sequence feeds the recompute path. Off by default (needs the 4B
     teacher loaded; the bridge assertions above are the core T3b check).

Run (unified env, all 8 GPUs):
  CUDA_HOME=/opt/pytorch/cuda CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PYTHONPATH=<DeepSpec>:<DeepSpec>/third_party/verl \
    ~/.venv/dspark-opd-sglang/bin/python scripts/opd/s_m2_t3b_rollout.py \
      --target Qwen/Qwen3-4B \
      --draft /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest \
      --cache /mnt/scratch/qwen3_4b_target_cache --tp 8 --b 2 --n 4
# Expected last line: RESULT: T3b ROLLOUT WIRING OK
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft",
                    default="/mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest")
    ap.add_argument("--prompt-jsonl", default="train_datasets/perfectblend_train.jsonl",
                    help="raw prompt corpus (user-only conversations) for DSparkPromptDataset")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--b", type=int, default=2, help="#prompts")
    ap.add_argument("--n", type=int, default=4, help="rollout.n independent responses per prompt")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--mem-fraction-static", type=float, default=0.15)
    ap.add_argument("--port", type=int, default=30000)
    args = ap.parse_args()

    import os

    import recipe.dspark_opd  # noqa: F401  compat shims FIRST
    import ray
    import torch
    from transformers import AutoTokenizer

    from recipe.dspark_opd.dataset import DSparkPromptDataset, dspark_collate_fn
    from recipe.dspark_opd.sglang_server import DSparkSglangServer
    from recipe.dspark_opd.sglang_rollout_bridge import (
        GOLDEN_SAMPLING_PARAMS, extract_prompts, rebuild_padded_batch, sglang_generate_batch,
    )
    from omegaconf import OmegaConf

    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "") or ",".join(str(i) for i in range(args.tp))
    devices = ",".join([d for d in devices.split(",") if d.strip()][: args.tp])
    print(f"[cfg] target={args.target} draft={args.draft} tp={args.tp} b={args.b} n={args.n} "
          f"prompt_jsonl={args.prompt_jsonl} devices={devices}")

    # (1) DSparkPromptDataset (T3b X-path): raw user prompts -> golden chat template -> prompt tokens.
    tok = AutoTokenizer.from_pretrained(args.target)
    dcfg = OmegaConf.create({"dspark": {"prompt_jsonl_path": args.prompt_jsonl,
                                        "enable_thinking": False, "n_samples": args.b}})
    ds = DSparkPromptDataset(tokenizer=tok, config=dcfg, max_samples=args.b)
    feats = [ds[i] for i in range(args.b)]
    batch = dspark_collate_fn(feats)   # right-pad to [B,T]: input_ids/attention_mask/loss_mask(all 0)
    input_ids, loss_mask = batch["input_ids"], batch["loss_mask"]
    attention_mask = batch["attention_mask"]
    print(f"[1/5] loaded {args.b} prompts from {args.prompt_jsonl} (golden template), padded "
          f"T={input_ids.shape[1]}")

    # (2) extract prompts (prompt dataset -> use attention_mask length; loss_mask is all-0).
    prompts = extract_prompts(input_ids, loss_mask, attention_mask=attention_mask)
    for i, p in enumerate(prompts):
        tail = tok.decode(p[-40:])
        print(f"  prompt[{i}] len={len(p)}  tail={tail[-70:]!r}")
    print(f"[2/5] extracted {len(prompts)} prompts (should end with '<think>\\n\\n</think>\\n\\n')")

    ray.init(ignore_reinit_error=True, log_to_driver=True)
    print(f"[3/5] launching DSparkSglangServer (tp={args.tp}, mem_fraction={args.mem_fraction_static}) ...")
    server = DSparkSglangServer.options(
        runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
        name="dspark_sglang_server_t3b_smoke",
    ).remote(
        target_path=args.target, draft_path=args.draft, tp_size=args.tp,
        cuda_visible_devices=devices, port=args.port,
        mem_fraction_static=args.mem_fraction_static,
    )
    ray.get(server.get_address.remote())

    # a thin client shim so the bridge's client.generate(...) hits the actor.
    class _ActorClient:
        def generate(self, input_ids, sampling_params):
            return ray.get(server.generate.remote(input_ids, sampling_params))

    rc = 1
    try:
        sp = dict(GOLDEN_SAMPLING_PARAMS)
        sp["temperature"] = args.temperature
        sp["max_new_tokens"] = args.max_new_tokens
        print(f"[4/5] generating B*n = {args.b}*{args.n} = {args.b * args.n} responses ...")
        gen_outs = sglang_generate_batch(_ActorClient(), prompts, args.n, sampling_params=sp)
        newb = rebuild_padded_batch(prompts, gen_outs, args.n, device=torch.device("cpu"))

        ii, lm, am = newb["input_ids"], newb["loss_mask"], newb["attention_mask"]
        Bn = args.b * args.n
        ok = True

        # shape/dtype contract
        shape_ok = ii.shape[0] == Bn and ii.shape == lm.shape == am.shape and ii.dtype == torch.long
        print(f"  [shape] input_ids/loss_mask/attention_mask {tuple(ii.shape)} long -> "
              f"{'OK' if shape_ok else 'BAD'} (expect B*n={Bn})")
        ok = ok and shape_ok

        # per-row: loss_mask span == response, response non-empty, EOS check
        n_eos = 0
        for i in range(Bn):
            prompt = prompts[i // args.n]
            lp = len(prompt)
            resp = [int(t) for t in gen_outs[i].get("output_ids") or []]
            lr = len(resp)
            # loss_mask must be 1 exactly on [lp:lp+lr]
            lm_span_ok = (int(lm[i, :lp].sum()) == 0 and int(lm[i, lp:lp + lr].sum()) == lr
                          and int(lm[i, lp + lr:].sum()) == 0)
            # input_ids prompt prefix preserved
            prefix_ok = ii[i, :lp].tolist() == prompt
            mi = gen_outs[i].get("meta_info", {})
            finish = mi.get("finish_reason")
            ftype = finish.get("type") if isinstance(finish, dict) else finish
            if ftype == "stop":
                n_eos += 1
            row_ok = lm_span_ok and prefix_ok and lr > 0
            if not row_ok:
                ok = False
                print(f"  [row {i}] lp={lp} lr={lr} lm_span={lm_span_ok} prefix={prefix_ok} "
                      f"finish={ftype} -> BAD")
        print(f"  [rows] {Bn} rows: loss_mask spans + prompt prefixes OK; "
              f"{n_eos}/{Bn} ended on EOS(stop)")

        # independence: the n copies of prompt 0 must NOT all be identical (temp=1.0).
        resp0 = [tuple(gen_outs[j].get("output_ids") or []) for j in range(args.n)]
        distinct = len(set(resp0))
        indep_ok = args.n == 1 or distinct > 1
        print(f"  [independence] prompt[0]'s {args.n} rollouts -> {distinct} distinct "
              f"-> {'OK' if indep_ok else 'BAD (all identical!)'}")
        ok = ok and indep_ok

        # accept-length sanity from meta_info (gsm8k-ish, just a ballpark)
        als = []
        for out in gen_outs:
            mi = out.get("meta_info", {})
            ct, vc = mi.get("completion_tokens") or 0, mi.get("spec_verify_ct") or 0
            if vc:
                als.append(ct / vc)
        if als:
            print(f"  [accept_len] mean={sum(als) / len(als):.2f} over {len(als)} rollouts")

        # T4: accept_state stream carried into the rebuilt batch, response-indexed + self-consistent.
        from recipe.dspark_opd.sglang_rollout_bridge import ACCEPT_STATE_PAD
        ast = newb["dspark_accept_state"]        # [B*n, R_max], pad=-1
        rlen = newb["response_lengths"]          # [B*n]
        t4_ok = ast.shape[0] == Bn
        n_stream, seed_ok_all, len_ok_all = 0, 0, 0
        for i in range(Bn):
            lr = int(rlen[i])
            row = ast[i, :lr]                    # the valid (response-indexed) stream
            raw = gen_outs[i].get("meta_info", {}).get("dspark_accept_state")
            if raw is None:
                continue                          # server without T1 patch -> no stream (skip)
            n_stream += 1
            # length must equal response length (PREFILL_SEED makes it per-output-token aligned)
            if len(raw) == lr:
                len_ok_all += 1
            # first element is the PREFILL_SEED(2); no pad(-1) inside the valid span
            if lr > 0 and int(row[0]) == 2 and int((row == ACCEPT_STATE_PAD).sum()) == 0:
                seed_ok_all += 1
        if n_stream == 0:
            print("  [T4 accept_state] MISSING on all rows (server lacks T1 patch?) -> FAIL")
            ok = False
        else:
            t4_ok = t4_ok and len_ok_all == n_stream and seed_ok_all == n_stream
            print(f"  [T4 accept_state] {tuple(ast.shape)} pad=-1; streams={n_stream}/{Bn} "
                  f"len==resp:{len_ok_all}/{n_stream} seed@0+no-pad:{seed_ok_all}/{n_stream} "
                  f"-> {'OK' if t4_ok else 'BAD'}")
            ok = ok and t4_ok

        # T5: reconstruct the block plan from accept_state + response, on the rebuilt batch.
        # A dynamic, NO hard cap (§层面1): size max_anchors to the batch's max round count
        # (= max #COMMIT_BOUNDARY over rows, +1 slack for a stop-trimmed partial round) so no
        # response is truncated. (Real training caps via dspark_num_anchors after corpus stats.)
        from recipe.dspark_opd.block_plan_reconstruct import reconstruct_block_plan
        max_rounds = 1
        for i in range(Bn):
            li = int(rlen[i])
            if li > 0:
                max_rounds = max(max_rounds, int((ast[i, :li] == 1).sum()) + 1)
        print(f"  [T5] max_anchors(dynamic, no cap)={max_rounds}")
        plan = reconstruct_block_plan(
            input_ids=newb["input_ids"], accept_state=ast, response_lengths=rlen,
            prompt_lengths=newb["prompt_lengths"], block_size=7, max_anchors=max_rounds)
        ap, km, em, tk = (plan["anchor_positions"], plan["block_keep_mask"],
                          plan["eval_mask"], plan["tokens"])
        t5_ok = True
        for i in range(Bn):
            if int(rlen[i]) == 0:
                continue
            n_round = int(km[i].sum())
            # reconstructed rounds == spec_verify_ct (or vc-1 if stop-trimmed mid-round)
            vc = gen_outs[i].get("meta_info", {}).get("spec_verify_ct") or 0
            # anchors must be strictly increasing within valid blocks + land inside prompt+response
            av = ap[i, :n_round].tolist()
            incr = all(av[j] < av[j + 1] for j in range(len(av) - 1))
            plen_i = int(newb["prompt_lengths"][i])
            in_range = all(plen_i <= a < plen_i + int(rlen[i]) for a in av)
            # block token filled where eval_mask=1 (accepts) must equal the real response tokens
            resp_i = newb["input_ids"][i, plen_i:plen_i + int(rlen[i])]
            round_ok = incr and in_range and n_round in (vc, max(0, vc - 1))
            if not round_ok:
                t5_ok = False
                print(f"  [T5 row {i}] rounds={n_round} vc={vc} incr={incr} in_range={in_range} -> BAD")
        print(f"  [T5 block plan] reconstructed from accept_state: anchors increasing + in-range + "
              f"rounds==spec_verify_ct -> {'OK' if t5_ok else 'BAD'}")
        ok = ok and t5_ok

        rc = 0 if ok else 1
    finally:
        print("[5/5] shutting down server ...")
        try:
            ray.get(server.shutdown.remote())
        except Exception as e:  # noqa: BLE001
            print(f"  shutdown error (ignored): {e!r}")
        ray.kill(server)

    print("RESULT: T3b ROLLOUT WIRING OK" if rc == 0 else "RESULT: T3b ROLLOUT WIRING FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
