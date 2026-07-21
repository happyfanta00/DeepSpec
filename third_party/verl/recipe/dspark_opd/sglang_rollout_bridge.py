"""DSpark-OPD Stage-M2 T3b — sglang rollout bridge (prompt extract -> generate -> rebuild).

The fused train_step's Phase-1 used to sample draft blocks INSIDE a fixed cached sequence
(dspark_block_rollout). T3b instead makes upstream DSPARK sglang generate a FRESH response
from the prompt, live. This module holds the pure glue so it can be unit-tested apart from
the worker (mirrors block_rollout.py / teacher_scoring.py):

  1. extract_prompts:   cached (input_ids, loss_mask) -> prompt token lists.
     The cache stores prompt+response with loss_mask=1 on response, 0 on prompt+padding
     (verified single-turn contiguous). prompt = input_ids[:first loss_mask==1] — ends at
     the assistant generation-prompt boundary (`<|im_start|>assistant\n`).
  2. sglang_generate_batch:  B prompts x rollout_n -> B*n live DSPARK responses.
     Uses EXPLICIT repeat (n single-sequence requests per prompt), NOT sglang's native n>1
     (unverified with spec-decode; explicit gives genuinely-independent temp=1.0 rollouts and
     matches the T2/T3a-validated single-request path). Feeds VARIABLE-LENGTH token lists (no
     padding — the engine self-batches), mirroring standard verl (sglang_rollout.py:690).
  3. rebuild_padded_batch:  variable-length (prompt+new_response) -> right-padded batch with
     input_ids / loss_mask (1 on new response) / attention_mask. We pad ourselves — standard
     verl does the same (sglang_rollout.py:223,765). target_hidden_states is recomputed by the
     caller (worker) via the co-resident teacher on the NEW sequence.

See docs/opd/dspark-on-sglang-design.md §5.6b.
"""
from __future__ import annotations

import torch

# eval golden caliber (§2.4): pure softmax multinomial, no top_p/top_k/min_p truncation.
GOLDEN_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "max_new_tokens": 2048,
}

# T1 draft-info stream (§5.4): sglang meta_info key + per-token codes (0=ACCEPT, 1=COMMIT_BOUNDARY,
# 2=PREFILL_SEED). Since 0 is a valid code, the padded accept_state tensor uses a distinct pad
# sentinel so T5 can tell padding from ACCEPT.
DSPARK_ACCEPT_STATE_KEY = "dspark_accept_state"
ACCEPT_STATE_PAD = -1


def extract_prompts(input_ids: torch.Tensor, loss_mask: torch.Tensor,
                    attention_mask: torch.Tensor | None = None) -> list[list[int]]:
    """Per row, the prompt token list (no padding) to feed sglang.

    Two source layouts:
      - PROMPT dataset (T3b default, DSparkPromptDataset): input_ids IS the golden prompt (incl.
        the enable_thinking=False <think></think> shell), loss_mask all-0, attention_mask marks the
        real prompt tokens. -> prompt = input_ids[:attention_mask.sum()] (right-padded to the batch).
      - CACHE dataset (legacy prompt+response conversation): loss_mask=1 on the response; the first
        loss_mask>0 index is the response start = generation-prompt boundary, so prompt =
        input_ids[:first loss_mask==1]. (NOTE: this drops the response's leading <think></think>
        shell into the response span — mismatches the golden prompt; prefer the PROMPT dataset.)
    Preference: if a row's loss_mask is all-0 (prompt dataset), use attention_mask / non-pad length;
    else cut at the first response token (cache).
    """
    B = input_ids.shape[0]
    prompts: list[list[int]] = []
    for b in range(B):
        lm = loss_mask[b]
        nz = torch.nonzero(lm > 0, as_tuple=False).flatten()
        if len(nz) > 0:
            end = int(nz[0].item())              # cache path: first response token
        elif attention_mask is not None:
            end = int(attention_mask[b].sum().item())   # prompt-dataset path: real prompt length
        else:
            nzt = torch.nonzero(input_ids[b] != 0, as_tuple=False).flatten()
            end = int(nzt[-1].item()) + 1 if len(nzt) > 0 else int(input_ids.shape[1])
        prompts.append([int(t) for t in input_ids[b, :end].tolist()])
    return prompts


def sglang_generate_batch(client, prompts: list[list[int]], rollout_n: int,
                          sampling_params: dict | None = None,
                          max_concurrency: int = 0) -> list[dict]:
    """Generate rollout_n independent responses per prompt via B*n single requests, fired
    CONCURRENTLY so the server's continuous batching runs many at once.

    `client` is a handle with .generate(input_ids, sampling_params) -> raw server dict (either
    the T3a DSparkSglangServer actor's .generate.remote wrapped by the caller, or an
    HttpServerEngineAdapter(launch_server=False)). Returns a flat list of B*n result dicts in
    row-major (prompt-major) order: [p0_r0, p0_r1, ..., p0_r{n-1}, p1_r0, ...] — matching
    repeat_interleave(n) semantics so the batch dim lines up with the rest of the pipeline.

    ⚡ CONCURRENCY (fixes #running-req==1): each client.generate is an INDEPENDENT blocking HTTP
    POST (requests.post via _make_request, no shared mutable state on the adapter), so we dispatch
    all B*n via a thread pool instead of a serial loop. The server then sees B*n in-flight requests
    and batches them (continuous batching) -> #running-req rises to whatever the KV budget allows.
    We keep the EXPLICIT-repeat design (B*n genuinely-independent single requests, the T2/T3a-
    validated spec-decode path — NOT sglang's native n>1) so semantics are byte-identical to the
    old serial loop; only dispatch parallelism changes. Order is preserved by indexed placement.
    NOTE: with a small resident KV pool (mem_fraction_static=0.15, §5.6b) the server caps how many
    of the B*n actually RUN at once (max_running_requests / free KV tokens) and queues the rest —
    still >> 1. Raise mem_fraction_static if rollout throughput dominates memory headroom.
    """
    sp = dict(sampling_params or GOLDEN_SAMPLING_PARAMS)
    n = int(rollout_n)
    # flat request list, prompt-major (row i -> prompt i//n) so the batch dim lines up downstream.
    reqs = [prompts[i // n] for i in range(len(prompts) * n)]
    total = len(reqs)
    if total <= 1:
        return [client.generate(input_ids=p, sampling_params=sp) for p in reqs]

    from concurrent.futures import ThreadPoolExecutor
    workers = int(max_concurrency) if max_concurrency and max_concurrency > 0 else total
    workers = max(1, min(workers, total))
    outs: list[dict] = [None] * total
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(client.generate, input_ids=reqs[i], sampling_params=sp): i
                for i in range(total)}
        for fut, i in futs.items():
            outs[i] = fut.result()      # re-raises any per-request error (fail loud, no silent drop)
    return outs


def _accept_state_of(out: dict) -> list[int]:
    """T4: pull the T1 per-output-token accept_state stream from one generate dict's meta_info.

    Returns [] if absent (e.g. server without the T1 patch). When present, its length equals the
    response length (len(output_ids)) — the T1 PREFILL_SEED makes it per-output-token aligned
    (§5.1). If a legacy stream is response_len-1 (no seed), the caller's length check catches it.
    """
    mi = out.get("meta_info") or {}
    st = mi.get(DSPARK_ACCEPT_STATE_KEY)
    return [int(s) for s in st] if st is not None else []


def rebuild_padded_batch(prompts: list[list[int]], gen_outs: list[dict], rollout_n: int,
                         *, device, pad_token_id: int = 0) -> dict:
    """Right-pad (prompt + new_response) into a batch; build loss_mask/attention_mask + carry the
    T1 accept_state stream (T4).

    gen_outs is the flat B*n list from sglang_generate_batch (prompt-major). For row i, the
    prompt is prompts[i // rollout_n] and the response is gen_outs[i]["output_ids"]. Produces:
      input_ids      [B*n, T_new] long   (prompt ++ response, right-padded with pad_token_id)
      loss_mask      [B*n, T_new] long   (1 over the response span, 0 on prompt + padding)
      attention_mask [B*n, T_new] long   (1 over real prompt+response tokens, 0 on padding)
      response_lengths [B*n] long        (len(output_ids) per row — the valid response span length)
      dspark_accept_state [B*n, R_max] long  (T4: the T1 stream per row, RESPONSE-indexed — index j
                                is response token j (== global position prompt_len+j); right-padded
                                with ACCEPT_STATE_PAD(-1). R_max = max response length. Empty/absent
                                streams -> all pad. len(stream)==response_length is asserted upstream.)
    T_new = max(len(prompt)+len(response)) across the batch. Mirrors standard verl's own
    right-padding of variable-length sglang outputs (sglang_rollout.py:223,765).
    """
    n = int(rollout_n)
    rows = []
    for i, out in enumerate(gen_outs):
        prompt = prompts[i // n]
        resp = [int(t) for t in (out.get("output_ids") or [])]
        rows.append((prompt, resp, _accept_state_of(out)))

    B = len(rows)
    T = max((len(p) + len(r)) for p, r, _ in rows) if rows else 0
    R = max((len(r) for _, r, _ in rows), default=0)
    input_ids = torch.zeros((B, T), dtype=torch.long, device=device)
    if pad_token_id != 0:
        input_ids.fill_(pad_token_id)
    loss_mask = torch.zeros((B, T), dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, T), dtype=torch.long, device=device)
    response_lengths = torch.zeros((B,), dtype=torch.long, device=device)
    prompt_lengths = torch.zeros((B,), dtype=torch.long, device=device)   # T5: response starts here
    accept_state = torch.full((B, R), ACCEPT_STATE_PAD, dtype=torch.long, device=device)
    for i, (prompt, resp, st) in enumerate(rows):
        lp, lr = len(prompt), len(resp)
        seq = prompt + resp
        n_real = lp + lr
        input_ids[i, :n_real] = torch.tensor(seq, dtype=torch.long, device=device)
        loss_mask[i, lp:lp + lr] = 1          # supervise the new response only
        attention_mask[i, :n_real] = 1
        response_lengths[i] = lr
        prompt_lengths[i] = lp
        if st:
            # T1 stream is response-indexed and (with the PREFILL_SEED) length == lr. Guard against
            # a short/legacy stream by clamping to lr so the tensor stays response-aligned.
            m = min(len(st), lr)
            accept_state[i, :m] = torch.tensor(st[:m], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "response_lengths": response_lengths,
        "prompt_lengths": prompt_lengths,
        "dspark_accept_state": accept_state,
    }
