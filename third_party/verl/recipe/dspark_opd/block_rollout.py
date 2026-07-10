"""DSpark block-parallel rollout core (IP-1, S2).

Pure function `dspark_block_rollout` that reuses the TRAINING block-parallel forward
building blocks (sample_anchor_positions / create_noise_embed / create_position_ids /
create_dspark_attention_mask / _forward_backbone / compute_logits) and then SAMPLES
draft tokens (markov head, in-block autoregressive) instead of teacher-forcing.

Kept as a plain function (no verl deps) so scripts/opd/s2_smoke.py can test it in
isolation. `DSparkRollout.generate_sequences` (rollout.py) wraps it for the verl path.

Output tensors follow docs/opd/tensor-contract.md §S2:
    tokens       [B, A, blk]   long   sampled draft tokens
    logp_draft   [B, A, blk]   float  logπ of sampled tokens (<=0, finite on valid blocks)
    base_logits  [B, A, blk, V] float pre-markov, pre-sampling logits (for eval consistency)
    anchor_positions [B, A]    long
    block_keep_mask  [B, A]    bool
    eval_mask    [B, A, blk]   bool
"""
from __future__ import annotations

import torch

from deepspec.modeling.dspark.common import (
    build_eval_mask,
    create_dspark_attention_mask,
    create_noise_embed,
    create_position_ids,
    sample_anchor_positions,
)


@torch.no_grad()
def dspark_block_rollout(
    model,
    *,
    input_ids: torch.Tensor,            # [B, T] long
    loss_mask: torch.Tensor,            # [B, T]
    target_hidden_states: torch.Tensor,  # [B, T, L*H]
    num_anchors: int,
    temperature: float = 1.0,
    top_k: int = 0,                                  # K: top-k dense candidates (0 = none)
    anchor_positions: torch.Tensor | None = None,   # [B, A] to force fixed anchors
    block_keep_mask: torch.Tensor | None = None,     # [B, A] paired with anchor_positions
) -> dict:
    device = input_ids.device
    bsz, seq_len = input_ids.shape
    block_size = int(model.block_size)

    # 1) anchors (sampled, or forced for reproducible cross-checks)
    if anchor_positions is None:
        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len, loss_mask=loss_mask, num_anchors=int(num_anchors), device=device,
        )
    assert block_keep_mask is not None
    num_blocks = anchor_positions.size(1)

    # 2) draft inputs: per block [anchor_token, mask, mask, ...]
    noise_embedding = create_noise_embed(
        model.embed_tokens, input_ids, anchor_positions, block_keep_mask,
        mask_token_id=int(model.mask_token_id), block_size=block_size,
    )
    # 3) position ids (context + draft) and block-diagonal mask
    context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
    draft_position_ids = create_position_ids(anchor_positions, block_size)
    full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
    dspark_attn_mask = create_dspark_attention_mask(
        anchor_positions=anchor_positions, block_keep_mask=block_keep_mask,
        seq_len=seq_len, block_size=block_size, device=device,
    )
    # 4) block-parallel backbone (context = target hidden as cross-attn K/V)
    output_hidden = model._forward_backbone(
        position_ids=full_position_ids,
        noise_embedding=noise_embedding,
        target_hidden_states=target_hidden_states,
        attention_mask=dspark_attn_mask,
    )
    output_hidden_4d = output_hidden.reshape(bsz, num_blocks, block_size, -1)

    # 5) invalid blocks (block_keep_mask False) attend to nothing -> NaN rows; zero them
    #    so logits stay finite. eval_mask excludes them downstream anyway.
    keep = block_keep_mask.view(bsz, num_blocks, 1, 1)
    output_hidden_4d = torch.where(keep, output_hidden_4d, torch.zeros_like(output_hidden_4d))

    # 6) base logits (pre-markov, pre-sampling) — exposed for eval-consistency check
    base_logits = model.compute_logits(output_hidden_4d)  # [B, A, blk, V]

    # 7) sample draft tokens (markov head does in-block autoregression), vectorized over B*A
    anchor_tokens = torch.gather(input_ids, 1, anchor_positions)  # [B, A]
    flat_base = base_logits.reshape(bsz * num_blocks, block_size, -1)
    flat_hidden = output_hidden_4d.reshape(bsz * num_blocks, block_size, -1)
    flat_first_prev = anchor_tokens.reshape(bsz * num_blocks)
    sampled_flat, corrected_flat = model.sample_draft_tokens(
        flat_base,
        first_prev_token_ids=flat_first_prev,
        temperature=float(temperature),
        hidden_states=flat_hidden,
    )
    tokens = sampled_flat.reshape(bsz, num_blocks, block_size)
    corrected_logits = corrected_flat.reshape(bsz, num_blocks, block_size, -1)

    # 8) logp of sampled tokens under (temperature-scaled) corrected logits
    temp = float(temperature) if float(temperature) > 0 else 1.0
    logp_all = torch.log_softmax(corrected_logits.float() / temp, dim=-1)
    logp_draft = logp_all.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)  # [B, A, blk]
    # zero-out invalid blocks (their logits were zeroed; keep logp finite & marked by eval_mask)
    logp_draft = torch.where(
        block_keep_mask.unsqueeze(-1), logp_draft, torch.zeros_like(logp_draft)
    )

    # 8b) top-k dense candidates (only_stu): take from the SAME markov-corrected dist as sampling
    #     (logp_all), so candidate set matches the sampling distribution. ids only; the grad
    #     logp is recomputed in S4. K=0 -> no candidates (single-sample OPD).
    student_top_k_ids = None
    student_top_k_logp = None
    if int(top_k) > 0:
        k = min(int(top_k), logp_all.shape[-1])
        student_top_k_logp, student_top_k_ids = torch.topk(logp_all, k, dim=-1)  # [B,A,blk,K]

    # 9) eval_mask (which block positions count) — same construction as training forward
    label_offsets = torch.arange(1, block_size + 1, device=device).view(1, 1, -1)
    label_indices = anchor_positions.unsqueeze(-1) + label_offsets
    safe_label_indices = label_indices.clamp(max=seq_len - 1)
    safe_label_indices = torch.where(
        block_keep_mask.unsqueeze(-1), safe_label_indices, torch.zeros_like(safe_label_indices)
    )
    eval_mask = build_eval_mask(
        seq_len=seq_len, loss_mask=loss_mask,
        label_indices=label_indices, safe_label_indices=safe_label_indices,
        block_keep_mask=block_keep_mask,
    )

    return {
        "tokens": tokens,
        "logp_draft": logp_draft,
        "student_top_k_ids": student_top_k_ids,      # [B,A,blk,K] or None (top_k=0)
        "student_top_k_logp": student_top_k_logp,    # [B,A,blk,K] or None
        "base_logits": base_logits,
        "anchor_positions": anchor_positions,
        "block_keep_mask": block_keep_mask,
        "eval_mask": eval_mask,
    }
