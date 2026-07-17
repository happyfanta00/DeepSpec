#!/usr/bin/env python3
"""Stage-M2 T1 CPU unit test: DSPARK draft-info export (single per-token stream).

T1 produces exactly one per-output-token stream in upstream sglang,
``dspark_accept_state``, with three codes:
  0 = ACCEPT          a drafted token accepted inside a decode block
  1 = COMMIT_BOUNDARY a decode round's trailing bonus/correction token
  2 = PREFILL_SEED    the prefill-produced first response token (block-0 anchor)

The seed makes the stream index-aligned with ``output_ids`` from position 0. That
alignment is load-bearing: the customized_info transport slices every value by
output-token offset, and stop/EOS trimming crops ``output_ids_through_stop`` by a
token position. Without the seed the stream was off by one (decode coords vs
output_ids coords), which was invisible on untrimmed sequences but silently
dropped the final round's boundary whenever a sequence was stop-trimmed. With the
seed the crop applies to the stream exactly as it does to output_ids, so
``len(accept_state) == completion_tokens`` holds *exactly*.

The design (docs/opd/dspark-on-sglang-design.md §5.4) originally called for a
*second* per-round ``dspark_block_anchors`` stream, but that stream does not
survive the transport (it is per-round, not per-token). Instead we keep the one
per-token stream and reconstruct the block anchors on the training side:
  - block 0's anchor is the SEED at index 0
  - every COMMIT_BOUNDARY is the last token of a decode block and anchors the next
  - a trailing non-boundary tail is a partial (stop-trimmed) block

This test locks in the encoding + the reconstruction, including the stop-trimming
case that the golden-caliber server run surfaced.

No GPU / server / torch needed. Import the *real* upstream symbols so this test
fails if the production encoding drifts.

Usage:
    ~/.venv/dspark-opd-sglang/bin/python scripts/opd/dspark_stream_unit.py
    # expected last line: RESULT: T1 STREAM UNIT OK
"""
from __future__ import annotations

import sys
from typing import List, Tuple

try:
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        DSPARK_ACCEPT_STATE_KEY,
        DSPARK_STATE_ACCEPT as ACCEPT,
        DSPARK_STATE_COMMIT_BOUNDARY as BOUNDARY,
        DSPARK_STATE_PREFILL_SEED as SEED,
        dspark_round_accept_state,
    )
except Exception as e:  # pragma: no cover - env guard
    print(f"FATAL: cannot import upstream T1 symbols ({type(e).__name__}: {e})")
    print("Run with the sglang env: ~/.venv/dspark-opd-sglang/bin/python")
    sys.exit(2)


# --- reference reconstruction (mirrors what T5 will do on the training side) ---


def reconstruct(accept_state: List[int]) -> Tuple[List[int], List[int]]:
    """From the per-token accept_state, recover (anchors, block_lengths).

    ``anchors[r]`` is the output_ids index of the token that anchors decode round
    ``r`` (the SEED for r=0, else the previous round's boundary token).
    ``block_lengths[r]`` is how many tokens round ``r`` committed (accepts + its
    bonus); the final block is partial when the sequence was stop-trimmed before
    its boundary. ``len(anchors) == spec_verify_ct``.
    """
    assert accept_state and accept_state[0] == SEED, "stream must start with SEED"
    assert accept_state.count(SEED) == 1, "exactly one SEED expected"

    boundaries = [i for i, s in enumerate(accept_state) if s == BOUNDARY]
    ends_with_boundary = accept_state[-1] == BOUNDARY

    # Each decode round ends at its boundary token; a stop-trimmed final round
    # ends at the last index without a boundary. Round r is anchored by the
    # previous round's end (the seed at 0 anchors round 0).
    ends = list(boundaries)
    if not ends_with_boundary:
        ends.append(len(accept_state) - 1)
    anchors = [0] + ends[:-1]
    block_lengths = [ends[r] - anchors[r] for r in range(len(ends))]
    return anchors, block_lengths


def simulate_stream(rounds: List[int], *, trim_last_to: int | None = None) -> List[int]:
    """Build a full stream: SEED + concatenated decode rounds.

    ``trim_last_to`` optionally crops the whole stream to that length to model
    stop/EOS trimming mid-round (the crop the transport applies via finished_len).
    """
    stream: List[int] = [SEED]
    for k in rounds:
        stream.extend(dspark_round_accept_state(k))
    if trim_last_to is not None:
        stream = stream[:trim_last_to]
    return stream


# --- checks ---


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_round_encoding() -> None:
    check(dspark_round_accept_state(3) == [ACCEPT, ACCEPT, ACCEPT, BOUNDARY],
          "round(3) must be [0,0,0,1]")
    check(dspark_round_accept_state(0) == [BOUNDARY], "round(0) must be [1]")
    check(dspark_round_accept_state(6) == [ACCEPT] * 6 + [BOUNDARY],
          "full-accept round encoding wrong")
    check(SEED not in (ACCEPT, BOUNDARY), "SEED code must be distinct")
    print("[1/5] round encoding + distinct SEED code ........... OK")


def test_reconstruction_matches_handcalc() -> None:
    # gamma=7: a full-accept round commits 6 drafts + 1 bonus = 7 tokens.
    rounds = [2, 6, 0, 4]
    stream = simulate_stream(rounds)
    anchors, block_lengths = reconstruct(stream)

    # One decode block per round; each commits k+1 tokens.
    expected_lengths = [k + 1 for k in rounds]
    check(block_lengths == expected_lengths,
          f"block lengths {block_lengths} != {expected_lengths}")

    # Anchors: seed(0), then each prior round's boundary position.
    # boundary of round r sits at 1(seed) + sum(len_0..r).
    expected_anchors = [0]
    pos = 1  # after seed
    for L in expected_lengths[:-1]:
        pos += L
        expected_anchors.append(pos - 1)  # boundary index = end of that block
    check(anchors == expected_anchors,
          f"anchors {anchors} != {expected_anchors}")
    print("[2/5] anchors/block-plan reconstruction vs hand-calc . OK")


def test_self_consistency_untrimmed() -> None:
    rounds = [6, 6, 3, 6, 0, 5, 6, 1, 6, 6, 2]
    stream = simulate_stream(rounds)

    spec_verify_ct = len(rounds)
    # completion = prefill token + all committed decode tokens.
    completion_tokens = 1 + sum(k + 1 for k in rounds)

    check(len(stream) == completion_tokens,
          f"len {len(stream)} != completion_tokens {completion_tokens}")
    check(stream[0] == SEED and stream.count(SEED) == 1, "seed structure")
    num_boundaries = sum(1 for s in stream if s == BOUNDARY)
    check(num_boundaries == spec_verify_ct,
          f"#boundaries {num_boundaries} != spec_verify_ct {spec_verify_ct}")
    anchors, _ = reconstruct(stream)
    check(len(anchors) == spec_verify_ct,
          f"#decode rounds {len(anchors)} != spec_verify_ct {spec_verify_ct}")
    accept_len = completion_tokens / spec_verify_ct
    print(f"[3/5] untrimmed self-consistency (len==completion, "
          f"#bnd==verify_ct={spec_verify_ct}, accept_len={accept_len:.3f}) OK")


def test_stop_trimmed_tail() -> None:
    # Model the golden-run failure: EOS lands mid-round, trimming that round's
    # boundary. spec_verify_ct still counts the round, so #boundaries == vct-1 and
    # the stream ends in a non-boundary partial block. len must still == completion.
    rounds = [6, 6, 4, 5]              # 4 full decode rounds
    full = simulate_stream(rounds)     # SEED + rounds, ends in BOUNDARY
    # Trim inside the last round (drop its boundary + last accepts): keep SEED +
    # rounds[:-1] + 2 accepts of the last round.
    keep = 1 + sum(k + 1 for k in rounds[:-1]) + 2
    stream = full[:keep]

    spec_verify_ct = len(rounds)        # engine counted all 4 rounds
    completion_tokens = keep            # completion == emitted (trimmed) length

    check(len(stream) == completion_tokens, "trimmed len must equal completion")
    check(stream[-1] == ACCEPT, "trimmed tail should end mid-block (non-boundary)")
    num_boundaries = sum(1 for s in stream if s == BOUNDARY)
    check(num_boundaries == spec_verify_ct - 1,
          f"trimmed #boundaries {num_boundaries} != vct-1 {spec_verify_ct - 1}")

    anchors, block_lengths = reconstruct(stream)
    # Reconstruction still recovers all 4 decode rounds (last one partial).
    check(len(anchors) == spec_verify_ct,
          f"reconstructed rounds {len(anchors)} != spec_verify_ct {spec_verify_ct}")
    check(block_lengths[-1] == 2, f"partial last block should be len 2, got {block_lengths[-1]}")
    check(sum(block_lengths) + 1 == len(stream), "blocks + seed must cover stream")
    print("[4/5] stop-trimmed tail (bnd==vct-1, rounds recovered) OK")


def test_length_invariant_exact() -> None:
    # The strong contract T4/T5 rely on: with the seed, stream length equals the
    # number of emitted output tokens exactly, trimmed or not.
    for rounds, trim in ([[3, 4, 5], None], [[6, 6], 1 + 7 + 3], [[2], None]):
        stream = simulate_stream(rounds, trim_last_to=trim)
        emitted = trim if trim is not None else 1 + sum(k + 1 for k in rounds)
        check(len(stream) == emitted, f"len {len(stream)} != emitted {emitted}")
        check(stream[0] == SEED, "must start with seed")
    print("[5/5] exact length invariant (len == completion_tokens) . OK")


def main() -> int:
    print(f"T1 stream key = {DSPARK_ACCEPT_STATE_KEY!r}  "
          f"(ACCEPT={ACCEPT}, BOUNDARY={BOUNDARY}, SEED={SEED})")
    test_round_encoding()
    test_reconstruction_matches_handcalc()
    test_self_consistency_untrimmed()
    test_stop_trimmed_tail()
    test_length_invariant_exact()
    print("RESULT: T1 STREAM UNIT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
