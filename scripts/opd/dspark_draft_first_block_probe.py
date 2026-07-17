#!/usr/bin/env python3
"""Stage-M2 T2 end-to-end: does swapping the DSPARK draft actually change its
predictions? Compare two real ckpts on the FIRST verify block only.

Why first-block-only (design decision, user 2026-07-17): speculative decoding is
lossless -- the emitted tokens are decided by the TARGET, so swapping the draft
does NOT change the output and accept length may look nearly identical. The signal
that truly distinguishes two drafts is the draft's own predicted distribution q.
And we look only at block 0, whose input is fully determined (anchor = the prefill
first response token + the target's prefill hidden states), so both drafts see
IDENTICAL context and any difference is purely the weights -- no divergence from
later sampling trajectories.

This is the behavioral counterpart to dspark_draft_update_probe.py: that probe
proves the *weights* swap over the T2 draft-only update path (checksums); this
probe quantifies *how much the draft's predictions change* between the two ckpts,
which is what you ultimately care about when pushing a trained draft back.

Method (golden caliber: seed=980406 / chat template / enable_thinking=False /
temp=1.0, aligned to eval.py, design §2.4):
  for each gsm8k prompt:
    1. target prefill -> target hidden states; sample the anchor (first response
       token) under a FIXED per-sample seed so both drafts get the SAME anchor.
    2. feed the same (anchor, target_hidden) into EACH draft's block-0 forward
       (forward_dspark_draft_block + compute_logits, reused from draft_ops).
    3. compare block-0 base logits/probs A(base) vs B(opd) per slot:
         - argmax divergence (fraction of the 7 slots where greedy token differs)
         - q on A's greedy token under A vs under B (how the prob moved)
         - symmetric KL per slot (nats), and top-1 prob per slot
We compare the per-slot BASE distribution (model.compute_logits over the block
hidden), which is deterministic given inputs -- the markov head's serial sampling
adds seeded randomness we deliberately avoid here.

Runs on 1 GPU, no server (reads the loaded weights directly, which is exactly what
a draft-only update changes). Use the sglang env (transformers 5.12.1 loads the
repo DSpark modeling, §7.1).

Usage:
    CUDA_VISIBLE_DEVICES=1 ~/.venv/dspark-opd-sglang/bin/python \
      scripts/opd/dspark_draft_first_block_probe.py \
        --target Qwen/Qwen3-4B \
        --draft-a /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest \
        --draft-b /mnt/scratch/checkpoints/deepspec/dspark_opd_qwen3_4b/step_16000 \
        --task gsm8k --n 16
    # prints per-slot divergence between the two drafts; large => update is
    # behaviorally meaningful. RESULT: DRAFTS DIFFER (or IDENTICAL).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch


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


def _bar(x: float, lo: float, hi: float, width: int = 24) -> str:
    frac = 0.0 if hi <= lo else max(0.0, min(1.0, (x - lo) / (hi - lo)))
    return "█" * int(round(frac * width)) + "░" * (width - int(round(frac * width)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-4B")
    ap.add_argument("--draft-a", required=True, help="baseline draft ckpt (e.g. block7 base)")
    ap.add_argument("--draft-b", required=True, help="updated draft ckpt (e.g. opd step_16000)")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=980406)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from deepspec.data.parser import encode_chat_messages
    from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block
    from deepspec.modeling.dspark.common import extract_context_feature
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
    from deepspec.utils import seed_all
    from deepspec.utils.sampling import logits_to_probs, sample_from_probs

    device = "cuda"
    print(f"[cfg] target={args.target}\n      A={args.draft_a}\n      B={args.draft_b}")
    print(f"[cfg] task={args.task} n={args.n} seed={args.seed} temp={args.temperature} "
          f"enable_thinking=False (golden caliber)")

    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    draft_a = Qwen3DSparkModel.from_pretrained(
        args.draft_a, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    draft_b = Qwen3DSparkModel.from_pretrained(
        args.draft_b, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    block_size = int(draft_a.block_size)
    assert block_size == int(draft_b.block_size), "block_size mismatch"
    layer_ids = draft_a.target_layer_ids

    rows = load_task(args.task, args.n, args.seed)

    @torch.no_grad()
    def block0_base_probs(draft, anchor_token, target_hidden, num_input_tokens):
        """Block-0 per-slot base draft probs [block_size, vocab] for one draft.

        Rebuilds the exact block-0 inputs the evaluator uses (_propose): anchor in
        slot 0, mask tokens elsewhere; cross-attends the target prefill hidden.
        """
        draft_input_ids = torch.full(
            (1, block_size), int(draft.mask_token_id), dtype=torch.long, device=device
        )
        draft_input_ids[:, 0] = anchor_token
        position_ids = torch.arange(num_input_tokens + block_size, device=device).unsqueeze(0)
        block_hidden = forward_dspark_draft_block(
            draft,
            draft_input_ids=draft_input_ids,
            position_ids=position_ids,
            past_key_values_draft=DynamicCache(),
            target_hidden_states=target_hidden,
            start=num_input_tokens,
            block_size=block_size,
        )
        base_logits = draft.compute_logits(block_hidden[:, :block_size, :])
        return logits_to_probs(base_logits, float(args.temperature))[0]  # [block, vocab]

    slot_kl_sym = torch.zeros(block_size, dtype=torch.float64)
    slot_argmax_diff = torch.zeros(block_size, dtype=torch.float64)
    slot_qa_on_a = torch.zeros(block_size, dtype=torch.float64)  # A's prob on A's greedy
    slot_qb_on_a = torch.zeros(block_size, dtype=torch.float64)  # B's prob on A's greedy
    n_used = 0

    for idx, row in enumerate(rows):
        # Per-sample seed exactly like base_evaluator (seed + global position).
        seed_all(int(args.seed) + idx)
        messages = [{"role": "user", "content": row["turns"][0]}]
        input_ids = encode_chat_messages(
            tok, messages, add_generation_prompt=True, enable_thinking=False
        ).to(device)
        num_input_tokens = input_ids.shape[1]

        with torch.no_grad():
            out = target(
                input_ids=input_ids,
                past_key_values=DynamicCache(),
                use_cache=True,
                output_hidden_states=True,
                logits_to_keep=1,
            )
            # anchor = first response token, sampled under the fixed seed above so
            # BOTH drafts branch from the identical anchor.
            anchor = sample_from_probs(logits_to_probs(out.logits, float(args.temperature)))
            target_hidden = extract_context_feature(out.hidden_states, layer_ids)

            pa = block0_base_probs(draft_a, anchor[:, 0], target_hidden, num_input_tokens)
            pb = block0_base_probs(draft_b, anchor[:, 0], target_hidden, num_input_tokens)

        pa64, pb64 = pa.double(), pb.double()
        eps = 1e-12
        kl_ab = (pa64 * ((pa64 + eps).log() - (pb64 + eps).log())).sum(-1)
        kl_ba = (pb64 * ((pb64 + eps).log() - (pa64 + eps).log())).sum(-1)
        slot_kl_sym += (kl_ab + kl_ba).cpu()

        a_greedy = pa64.argmax(-1)
        b_greedy = pb64.argmax(-1)
        slot_argmax_diff += (a_greedy != b_greedy).double().cpu()
        slot_qa_on_a += pa64.gather(-1, a_greedy[:, None])[:, 0].cpu()
        slot_qb_on_a += pb64.gather(-1, a_greedy[:, None])[:, 0].cpu()
        n_used += 1
        if idx < 3:
            d = int((a_greedy != b_greedy).sum())
            print(f"  [{idx}] anchor={int(anchor[0,0])} slot0 KLsym={float(kl_ab[0]+kl_ba[0]):.4f} "
                  f"argmax-diff slots={d}/{block_size}")

    kl = (slot_kl_sym / n_used)
    amd = (slot_argmax_diff / n_used)
    qa = (slot_qa_on_a / n_used)
    qb = (slot_qb_on_a / n_used)

    print(f"\n=== first-block draft divergence: A(base) vs B(opd), n={n_used} ===")
    print(f"  {'slot':>4} {'sym-KL(nats)':>13} {'argmax-diff':>12}  "
          f"{'qA(argmaxA)':>11} {'qB(argmaxA)':>11}")
    klmax = float(kl.max()) if n_used else 1.0
    for s in range(block_size):
        print(f"  {s:>4} {float(kl[s]):>13.4f} {float(amd[s]):>11.1%}  "
              f"{float(qa[s]):>11.3f} {float(qb[s]):>11.3f}  {_bar(float(kl[s]), 0.0, klmax)}")
    mean_kl = float(kl.mean())
    mean_amd = float(amd.mean())
    print(f"\n  mean sym-KL over slots = {mean_kl:.4f} nats")
    print(f"  mean argmax divergence = {mean_amd:.1%}")
    print(f"  interpretation: q on A's greedy token drops {float(qa.mean()):.3f} -> "
          f"{float(qb.mean()):.3f} when swapping A->B (draft became {'different' if mean_kl>1e-3 else '~identical'}).")

    # A tiny threshold: floating-point noise between identical weights is ~0.
    differ = mean_kl > 1e-3 or mean_amd > 0.01
    print("RESULT: DRAFTS DIFFER (draft-only update is behaviorally meaningful)" if differ
          else "RESULT: DRAFTS ~IDENTICAL (unexpected for two distinct ckpts)")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "config": vars(args),
            "n": n_used,
            "slot_sym_kl": kl.tolist(),
            "slot_argmax_diff": amd.tolist(),
            "slot_qa_on_a": qa.tolist(),
            "slot_qb_on_a": qb.tolist(),
            "mean_sym_kl": mean_kl,
            "mean_argmax_diff": mean_amd,
        }, indent=2))
        print(f"  wrote {args.out}")
    return 0 if differ else 1


if __name__ == "__main__":
    raise SystemExit(main())
