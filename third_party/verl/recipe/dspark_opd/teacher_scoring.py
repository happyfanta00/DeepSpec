"""Teacher (target) merged forward for the sglang DSPARK path.

`teacher_causal_hidden_and_topk`: ONE plain-causal teacher forward over the real (prompt++response)
sequence yields BOTH (a) the mid-layer hidden (the draft's cross-attn context) and (b) the teacher
top-K logπ at the block query positions. Valid because the block tokens are the REAL accepted
trajectory, so a causal forward's logits at global pos p=anchor+j equal the block-diagonal
block-query logits (same token, causal prefix 0..p, RoPE pos). See docs/opd/t5-train-step-tensor-contract.md.
"""
from __future__ import annotations

import torch


def _teacher_backbone(target):
    """Backbone (has .layers) of a (possibly FSDP-wrapped) HF CausalLM. Qwen3: target.model."""
    m = getattr(target, "_fsdp_wrapped_module", target)  # unwrap FSDP root if present
    return getattr(m, "model", m)


@torch.no_grad()
def teacher_causal_hidden_and_topk(
    target, *, input_ids, attention_mask, target_layer_ids,
    anchor_positions, block_keep_mask, block_size, teacher_top_k,
):
    """T5 MERGED teacher forward — ONE plain-causal forward yields BOTH:
      (a) target_hidden_states [B,T,L*H] (mid-layer hidden, draft cross-attn context), and
      (b) teacher top-k (ids + true logπ) at the block query positions [B,A,blk,Kt].

    Replaces the T5 path's two separate teacher forwards (recompute_target_hidden_states +
    score_blocks_flat). Valid ONLY on the sglang/T5 path where block tokens are the REAL accepted
    trajectory: block query (r,j) sits at global position p = anchor_r + j on the real sequence, so
    a plain-causal forward's logits at position p equal the block-diagonal forward's block-query
    logits (same token, same causal prefix 0..p, same RoPE pos). See t5-train-step-tensor-contract.md.

    FSDP note: teacher is FULL_SHARD, so params are gathered only inside target(...) and resharded
    after — we therefore CANNOT call lm_head separately. We do ONE full target(output_hidden_states=
    True) forward: capture mid layers via hooks + read the final .logits at the block positions.
    The [B,T,V] logits peak is bounded by the CALLER micro-batching this over the batch dim (same as
    the old recompute forward), and this is now the ONLY teacher forward on the T5 path.

    Args (single micro-batch already sliced by the caller):
      input_ids       [B,T]      right-padded (prompt++response)
      attention_mask  [B,T]      1 on real tokens
      target_layer_ids[L]        mid layers to capture for the draft context
      anchor_positions[B,A]      block anchor global index (p = anchor + j)
      block_keep_mask [B,A]      valid blocks
      block_size      int (blk)
      teacher_top_k   int (Kt)

    Returns dict:
      target_hidden_states [B,T,L*H] bf16
      t_ids                [B,A,blk,Kt] long   (teacher top-Kt ids at block positions; 0 on invalid)
      t_logp               [B,A,blk,Kt] float  (true full-vocab logπ = logit − logsumexp(V); 0 invalid)
    """
    backbone = _teacher_backbone(target)
    layer_modules = backbone.layers
    lids = [int(x) for x in target_layer_ids]
    B, A = int(anchor_positions.shape[0]), int(anchor_positions.shape[1])
    blk, Kt = int(block_size), int(teacher_top_k)
    T = int(input_ids.shape[1])
    device = input_ids.device

    cap: dict[int, torch.Tensor] = {}
    handles = []

    def _mk(lid):
        def _hook(_module, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            cap[lid] = t.detach()
        return _hook

    try:
        for lid in lids:
            assert lid >= 0, f"merged teacher supports real decoder layers only, got {lid}"
            handles.append(layer_modules[lid].register_forward_hook(_mk(lid)))
        out = target(input_ids=input_ids, attention_mask=attention_mask,
                     use_cache=False, output_hidden_states=False)
        # (a) mid-layer hidden for the draft cross-attn context (ascending-layer cat order).
        target_hidden_states = torch.cat([cap[lid] for lid in lids], dim=-1)   # [B,T,L*H]

        # (b) teacher top-Kt at block query positions p[b,r,j] = anchor[b,r] + j (clamped to T-1).
        logits = out.logits                                                    # [B,T,V]
        j_idx = torch.arange(blk, device=device).view(1, 1, blk)               # [1,1,blk]
        p = (anchor_positions.unsqueeze(-1) + j_idx).clamp_(max=T - 1)         # [B,A,blk] global pos
        p_flat = p.reshape(B, A * blk)                                         # [B,A*blk]
        gathered = torch.gather(                                               # [B,A*blk,V]
            logits, 1, p_flat.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))
        gathered = gathered.reshape(B, A, blk, -1).float()                    # [B,A,blk,V]
        kk = min(Kt, gathered.shape[-1])
        lse = torch.logsumexp(gathered, dim=-1, keepdim=True)                 # [B,A,blk,1]
        top_logits, t_ids = torch.topk(gathered, kk, dim=-1)                  # [B,A,blk,kk]
        t_logp = top_logits - lse                                            # true full-vocab logπ
        keep4 = block_keep_mask.view(B, A, 1, 1)
        t_logp = torch.where(keep4, t_logp, torch.zeros_like(t_logp))
        t_ids = torch.where(keep4, t_ids, torch.zeros_like(t_ids))
        return {"target_hidden_states": target_hidden_states, "t_ids": t_ids, "t_logp": t_logp}
    finally:
        for h in handles:
            h.remove()
        cap.clear()

