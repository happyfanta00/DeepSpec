"""DSpark-OPD Stage-M2 T5 — reconstruct the block plan from the accept_state stream.

T5 replaces the fused train_step's Phase-1 (dspark_block_rollout, which sampled draft blocks at
RANDOM anchors) with the REAL speculative-decoding trajectory that sglang actually walked, recovered
from the T1 `dspark_accept_state` stream (carried by T4). This module holds the pure reconstruction
so it is CPU-unit-testable apart from the worker (mirrors block_rollout.py / teacher_scoring.py).

Given, per sequence, the response-indexed accept_state (codes 0=ACCEPT, 1=COMMIT_BOUNDARY,
2=PREFILL_SEED; §5.4) and the response tokens, we recover the block plan and produce the SAME output
contract dspark_block_rollout used, so Phase-2 (score_blocks_flat) and Phase-3 (module(...)) run
UNCHANGED on the reconstructed blocks:

    anchor_positions [B, A]        long   input_ids index of each block's anchor
    block_keep_mask  [B, A]        bool   valid blocks (short seqs -> trailing False)
    eval_mask        [B, A, blk]   bool   in-block positions that count in the loss
    tokens           [B, A, blk]   long   the block's tokens (real accepted response tokens)

Block r covers response positions anchor_r+1 .. anchor_r+block_lengths[r] (§draft_ops slot i ->
anchor+1+i). Reconstruction of (anchors, block_lengths) reuses dspark_stream_unit.reconstruct
(T1-validated, incl. stop-trim).

★ Decisions locked (design §5.8, user 2026-07-17):
 - block internal token = the REAL accepted response tokens (true trajectory, not resampled).
 - variable accept length L_r is expressed by eval_mask (first L_r positions =1), tensor stays
   fixed-width [B,A,block_size]; L_r <= block_size always.
 - REJECT position (boundary, slot L_r) = OPTION B: eval_mask=0 there (its q/p are recorded by the
   forward for future experiments but do NOT enter the current loss).
 - DISCARDED tail (slots L_r+1..) = eval_mask 0 (draft's rejected proposals never left sglang;
   out of scope).

A = number of decode rounds (== spec_verify_ct), variable per sequence -> padded to the batch max
(capped at max_anchors; overflow rounds dropped with a loud log).
"""
from __future__ import annotations

import torch

# accept_state codes (mirror scripts/opd/dspark_stream_unit.py / T1 §5.4).
STATE_ACCEPT = 0
STATE_BOUNDARY = 1
STATE_SEED = 2


def _reconstruct_one(accept_state: list[int]) -> tuple[list[int], list[int], list[bool]]:
    """(anchors, block_lengths, has_boundary) from one response-indexed accept_state list.

    anchors[r] = OUTPUT/response index of round r's anchor (SEED@0 for r=0, else the previous
    round's boundary index). block_lengths[r] = tokens committed that round (accepts + its bonus/
    correction boundary); the final round is partial (no boundary) when stop-trimmed.
    has_boundary[r] = whether round r ended on a COMMIT_BOUNDARY (True) vs a stop-trimmed partial
    tail (False) — this distinguishes the reject slot (present iff the round has a boundary) from
    the trailing partial round that has no reject/correction token (Draft-OPD dual-stream).
    Identical anchors/lengths logic to dspark_stream_unit.reconstruct — inlined here to avoid a
    scripts/ import in the worker/Ray path.
    """
    if not accept_state or accept_state[0] != STATE_SEED:
        raise ValueError("accept_state must start with PREFILL_SEED(2)")
    boundaries = [i for i, s in enumerate(accept_state) if s == STATE_BOUNDARY]
    ends = list(boundaries)
    has_boundary = [True] * len(boundaries)
    if accept_state[-1] != STATE_BOUNDARY:
        ends.append(len(accept_state) - 1)   # stop-trimmed partial final round
        has_boundary.append(False)           # ...no boundary/reject token on this tail round
    anchors = [0] + ends[:-1]
    block_lengths = [ends[r] - anchors[r] for r in range(len(ends))]
    return anchors, block_lengths, has_boundary


def reconstruct_block_plan(
    *,
    input_ids: torch.Tensor,          # [B, T] rebuilt (prompt ++ response), right-padded
    accept_state: torch.Tensor,       # [B, R] response-indexed T1 stream, pad = ACCEPT_STATE_PAD(-1)
    response_lengths: torch.Tensor,   # [B] len(output_ids) per row
    prompt_lengths: torch.Tensor,     # [B] prompt token count per row (response starts here)
    block_size: int,
    max_anchors: int,
) -> dict:
    """Reconstruct dspark_block_rollout's output contract from the real trajectory.

    Per sequence b: slice the valid stream accept_state[b, :response_lengths[b]], reconstruct
    (anchors_out, block_lengths, has_boundary); each round r is one block. Map response index ->
    input_ids index via prompt_lengths[b]. Fill anchor/tokens/slot_type; cap rounds at max_anchors.

    slot_type[b,r,j] encodes the Draft-OPD DUAL STREAM per in-block slot (n_acc = block_lengths-1):
      -  0 (ACCEPT): slots 0..n_acc-1 — draft proposals that were accepted.
      -  1 (REJECT): slot n_acc — the boundary/correction token (present iff the round ended on a
                     COMMIT_BOUNDARY; the stop-trimmed final round has no boundary -> no reject slot).
      - -1 (NONE):   discarded tail (>n_acc) and padding — never enters the loss.
    Both accept & reject tokens live in the response, so their draft/teacher distributions are
    recomputable by the teacher-force forward (no hook, no rejected-proposal id needed).
    eval_mask = (slot_type >= 0) is returned too for back-compat (accept ∪ reject).
    """
    device = input_ids.device
    B = int(input_ids.shape[0])
    blk = int(block_size)
    A = int(max_anchors)

    anchor_positions = torch.zeros((B, A), dtype=torch.long, device=device)
    block_keep_mask = torch.zeros((B, A), dtype=torch.bool, device=device)
    slot_type = torch.full((B, A, blk), -1, dtype=torch.int8, device=device)  # -1=none/pad
    tokens = torch.zeros((B, A, blk), dtype=torch.long, device=device)

    n_overflow = 0
    for b in range(B):
        rlen = int(response_lengths[b])
        plen = int(prompt_lengths[b])
        if rlen <= 0:
            continue
        st = [int(x) for x in accept_state[b, :rlen].tolist()]
        anchors_out, block_lengths, has_boundary = _reconstruct_one(st)
        n_round = len(anchors_out)
        if n_round > A:
            n_overflow += n_round - A
            n_round = A
        resp = input_ids[b, plen:plen + rlen]     # [rlen] the response tokens (== output_ids)
        for r in range(n_round):
            a_out = anchors_out[r]                 # response index of this block's anchor
            blen = block_lengths[r]                # committed tokens this round (accepts + boundary)
            # anchor in input_ids coords. anchor token = response[a_out] (or the prompt's last token
            # for round 0's SEED, which is response[0]-1 conceptually — but a_out=0 IS response[0],
            # the SEED token itself, which anchors round 0; that matches draft_ops: block predicts
            # anchor+1.. so anchor = the token BEFORE the first predicted one).
            anchor_positions[b, r] = plen + a_out
            block_keep_mask[b, r] = True
            # in-block slots predict response[a_out+1 .. a_out+blen]; fill tokens with those reals.
            L = min(blen, blk)                     # committed count clamped to block_size
            for j in range(L):
                pos = a_out + 1 + j
                if pos < rlen:
                    tokens[b, r, j] = int(resp[pos])
            # DUAL STREAM: first n_acc slots = ACCEPT(0); the boundary slot (index n_acc) = REJECT(1)
            # iff this round ended on a real COMMIT_BOUNDARY (not a stop-trimmed partial tail).
            n_acc = max(0, min(blen - 1, blk))
            if n_acc > 0:
                slot_type[b, r, :n_acc] = 0                       # ACCEPT stream
            if has_boundary[r] and n_acc < blk:
                slot_type[b, r, n_acc] = 1                        # REJECT stream (one boundary slot)

    if n_overflow > 0:
        print(f"[reconstruct_block_plan] WARNING dropped {n_overflow} rounds exceeding "
              f"max_anchors={A} (increase dspark_num_anchors if this is frequent)", flush=True)

    return {
        "anchor_positions": anchor_positions,
        "block_keep_mask": block_keep_mask,
        "slot_type": slot_type,                        # [B,A,blk] int8: 0=accept 1=reject -1=none
        "eval_mask": (slot_type >= 0),                 # [B,A,blk] bool: accept ∪ reject (back-compat)
        "tokens": tokens,
    }
