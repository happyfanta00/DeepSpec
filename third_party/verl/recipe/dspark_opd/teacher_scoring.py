"""Teacher (target) block-diagonal scoring core (IP-4, S3, route A).

Given a batch's real context + the draft's per-anchor sampled blocks, compute the TARGET
model's log-prob on the student's top-K candidate tokens at each block position, conditioned
on that block's own sampled prefix ([real ctx 0..anchor] + [ỹ_1..ỹ_{k-1}]).

Two implementations (tensor-contract §S3):
  - score_blocks_reference : per-anchor loop, each block scored by an independent target
    forward (context 0..anchor prefill + [anchor_tok, ỹ_1..ỹ_blk], causal). Mirrors eval's
    verify_draft_tokens. OBVIOUSLY correct; used as the equivalence ground-truth.
  - score_blocks_flat : all anchors in ONE target forward via a flattened sequence
    [context(T) | A*blk block queries] + a 4D additive mask enforcing block-diagonal
    (anchor isolation) + within-block causal + context-up-to-anchor, with real position_ids.

S3 smoke asserts the two agree (the "flatten equivalence" golden check).

Output: logp_target_on_topk [B, A, blk, K] — teacher logπ on student_top_k_ids.
"""
from __future__ import annotations

import torch

NEG = float("-inf")


def _teacher_backbone(target):
    """Backbone (has .layers) of a (possibly FSDP-wrapped) HF CausalLM. Qwen3: target.model."""
    m = getattr(target, "_fsdp_wrapped_module", target)  # unwrap FSDP root if present
    return getattr(m, "model", m)


@torch.no_grad()
def recompute_target_hidden_states(target, *, input_ids, attention_mask, target_layer_ids):
    """Re-run the TEACHER over input_ids and capture the intermediate hidden states at
    target_layer_ids, concatenated on the feature dim — reproducing what prepare_target_cache.py
    stored as `target_hidden_states` (optimization #5: recompute instead of read from cache/dispatch).

    Mirrors prepare_target_cache.run_target_forward_with_hooks EXACTLY (same decoder-layer forward
    hooks, same cat order = target_layer_ids ascending) so the draft's cross-attn context is
    identical in layout. Differences vs the cached tensor are only: (a) bf16 vs the cache's fp8
    storage (recompute is MORE precise), (b) batch-invariance noise from right-padding (~1e-3, same
    order as S2/S3 flatten-equivalence). attention_mask (1 over real tokens) masks right-padding so
    padded positions never leak into real-token hidden.

    Returns target_hidden_states [B, T, len(target_layer_ids)*H] in the model's dtype (bf16).
    """
    backbone = _teacher_backbone(target)
    layer_modules = backbone.layers
    lids = [int(x) for x in target_layer_ids]
    cap: dict[int, torch.Tensor] = {}
    handles = []

    def _mk(lid):
        def _hook(_module, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            cap[lid] = t.detach()
        return _hook

    try:
        for lid in lids:
            assert lid >= 0, f"recompute supports real decoder layers only, got layer id {lid}"
            handles.append(layer_modules[lid].register_forward_hook(_mk(lid)))
        target(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
               output_hidden_states=False)
        return torch.cat([cap[lid] for lid in lids], dim=-1)  # [B,T,L*H], ascending-layer order
    finally:
        for h in handles:
            h.remove()
        cap.clear()


@torch.no_grad()
def _target_block_logits_reference(target, input_ids_row, anchor, blk, block_tokens):
    """One block's target logits [blk, V], mirroring eval verify_draft_tokens.

    input_ids_row: [T] real tokens (one sample). anchor: int. block_tokens: [blk] sampled ỹ.
    context = input_ids_row[:anchor+1] (real prefix incl. anchor); then append ỹ_1..ỹ_{blk-1}
    as the causal continuation. Positions anchor..anchor+blk. logits[j] = teacher dist that
    predicts the token at position anchor+1+j (i.e. scores ỹ_{j+1}).
    """
    device = input_ids_row.device
    # verify_input_ids = [anchor_tok, ỹ_1, ..., ỹ_{blk-1}]  (length blk; predicts ỹ_1..ỹ_blk)
    verify_ids = torch.cat([input_ids_row[anchor:anchor + 1], block_tokens[:blk - 1]], dim=0)
    ctx = input_ids_row[:anchor]                       # [anchor] real context before anchor
    full = torch.cat([ctx, verify_ids], dim=0).unsqueeze(0)  # [1, anchor+blk]
    pos = torch.arange(full.shape[1], device=device).unsqueeze(0)
    out = target(input_ids=full, position_ids=pos, use_cache=False)
    logits = out.logits[0]                             # [anchor+blk, V]
    # block query positions are the last blk positions; logits there predict ỹ_1..ỹ_blk
    return logits[anchor:anchor + blk]                 # [blk, V]


@torch.no_grad()
def score_blocks_reference(target, *, input_ids, tokens, anchor_positions, block_keep_mask,
                           student_top_k_ids):
    """Reference: per-block independent forward. Returns logp_target_on_topk [B,A,blk,K]."""
    B, A, blk = tokens.shape
    K = student_top_k_ids.shape[-1]
    device = input_ids.device
    out = torch.zeros(B, A, blk, K, device=device, dtype=torch.float32)
    for b in range(B):
        for i in range(A):
            if not bool(block_keep_mask[b, i]):
                continue
            a = int(anchor_positions[b, i])
            logits = _target_block_logits_reference(
                target, input_ids[b], a, blk, tokens[b, i])       # [blk, V]
            logp = torch.log_softmax(logits.float(), dim=-1)      # [blk, V]
            out[b, i] = logp.gather(-1, student_top_k_ids[b, i])  # [blk, K]
    return out


@torch.no_grad()
def score_blocks_flat(target, *, input_ids, tokens, anchor_positions, block_keep_mask,
                      student_top_k_ids):
    """All B samples × all A blocks in ONE batched forward (batch dim B + 4D block-diag mask).

    Per sample the sequence is [ctx: real input_ids (T)] ++ [A*blk block queries]; we stack the
    B samples into the batch dim so the FULL_SHARD teacher all-gathers its params ONCE (vs once
    per sample in the old per-sample loop — the big teacher-phase speedup). Batched attention
    isolates samples naturally (different batch rows never attend each other), so this is
    numerically identical to per-sample scoring (verified by the S3 batch-equivalence smoke).

    Block query flat index q = i*blk + j (anchor i, in-block pos j) has:
      - real position_id = anchor_i + j        (RoPE by true position, §S3 cond b)
      - token = [anchor_tok, ỹ_1..ỹ_{blk-1}] for that block (predicts ỹ_1..ỹ_blk)
    4D additive mask (per sample, query over A*blk, key over T + A*blk):
      - context keys: allowed if key_pos < anchor_i (causal up to but excluding anchor)
      - block keys: allowed iff same anchor (block-diagonal) AND in-block causal (j' <= j)
      - padding: excluded implicitly — anchors live in real tokens (loss_mask>0), so
        key_pos < anchor_i can never select a padded ctx column [real_len, T).
    """
    B, A, blk = tokens.shape
    K = student_top_k_ids.shape[-1]
    T = input_ids.shape[1]
    device = input_ids.device
    Q = A * blk
    mdtype = next(target.parameters()).dtype
    neg = torch.finfo(mdtype).min

    # ---- block query tokens [B,A,blk] = [anchor_tok, ỹ_1..ỹ_{blk-1}] per block ----
    anchor_tok = torch.gather(input_ids, 1, anchor_positions.clamp(max=T - 1))  # [B,A]
    blk_q_tokens = torch.cat([anchor_tok.unsqueeze(-1), tokens[:, :, : blk - 1]], dim=2)  # [B,A,blk]
    q_tokens = blk_q_tokens.reshape(B, Q)                                       # [B,Q]
    full_ids = torch.cat([input_ids, q_tokens], dim=1)                          # [B, T+Q]

    # ---- position ids: ctx = arange(T); block q = anchor_i + j (RoPE by true position) ----
    j_idx = torch.arange(blk, device=device).view(1, 1, blk).expand(B, A, blk)  # [B,A,blk]
    q_pos = (anchor_positions.unsqueeze(-1) + j_idx).reshape(B, Q)              # [B,Q]
    ctx_pos = torch.arange(T, device=device).view(1, T).expand(B, T)           # [B,T]
    full_pos = torch.cat([ctx_pos, q_pos], dim=1)                              # [B, T+Q]

    # ---- 4D additive mask [B,1,T+Q,T+Q]. Rows 0..T-1 = ctx (standard causal), rows T.. = block
    #      queries with block-diagonal + in-block-causal + ctx-up-to-anchor. ----
    row_anchor_id = torch.arange(A, device=device).repeat_interleave(blk)       # [Q] block id per query row
    row_j = j_idx[0].reshape(Q)                                                 # [Q] in-block pos per row
    row_anchor_pos = anchor_positions[:, row_anchor_id]                         # [B,Q] anchor pos per query row
    key_ctx_pos = torch.arange(T, device=device)                               # [T]
    # context keys: STRICTLY before anchor (anchor token is supplied as block-key j'=0, not ctx;
    # `<=` would double-count it — see S3 golden check). Also excludes padding (see docstring).
    allow_ctx = key_ctx_pos.view(1, 1, T) < row_anchor_pos.unsqueeze(-1)        # [B,Q,T]
    # block keys: same anchor (block-diagonal) AND in-block causal (key_j <= query_j). B-independent.
    same_block = row_anchor_id.view(Q, 1) == row_anchor_id.view(1, Q)          # [Q,Q]
    causal_in_block = row_j.view(1, Q) <= row_j.view(Q, 1)                     # [Q,Q]
    allow_blk = (same_block & causal_in_block).unsqueeze(0).expand(B, Q, Q)    # [B,Q,Q]
    allow_q = torch.cat([allow_ctx, allow_blk], dim=2)                         # [B,Q,T+Q]

    big = torch.zeros(B, 1, T + Q, T + Q, device=device, dtype=mdtype)
    ctx_causal = torch.triu(torch.full((T, T), neg, device=device, dtype=mdtype), diagonal=1)
    big[:, 0, :T, :T] = ctx_causal                                            # ctx rows: causal
    big[:, 0, :T, T:] = neg                                                   # ctx never attends block toks
    big[:, 0, T:, :] = torch.where(allow_q, torch.zeros((), dtype=mdtype, device=device),
                                   torch.full((), neg, dtype=mdtype, device=device))  # block-query rows

    model_out = target(input_ids=full_ids, position_ids=full_pos,
                       attention_mask=big, use_cache=False)
    logits = model_out.logits[:, T:]                                          # [B, Q, V] block-query logits
    logp = torch.log_softmax(logits.float(), dim=-1).reshape(B, A, blk, -1)   # [B,A,blk,V]
    gathered = logp.gather(-1, student_top_k_ids)                             # [B,A,blk,K]
    out = torch.where(block_keep_mask.view(B, A, 1, 1), gathered, torch.zeros_like(gathered))
    return out
