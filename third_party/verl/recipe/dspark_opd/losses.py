"""DSpark-OPD self-contained loss operators + orchestration (Step-1 decoupling).

Replaces the reward→advantage→PPO-policy-gradient path (build_opd_reward +
token_reward_direct + compute_policy_loss_vanilla) with differentiable loss operators that
backprop directly. See docs/opd/loss-refactor-design.md.

Architecture (mirrors Draft-OPD's registry pattern, adapted to DSpark's 4D block tensors):
  - register_dspark_loss(name): decorator -> DSPARK_LOSS_REGISTRY.
  - Operator signature: (ctx: DSparkLossContext, cfg) -> (per_token_loss [B,A,blk], metrics).
    Operators return the PER-TOKEN loss (NOT a scalar) and do NOT pre-multiply the decay/eval
    mask — aggregation + decay weighting happen ONCE in compose_dspark_loss.
  - compose_dspark_loss: weighted sum of each enabled term's decay-weighted token-mean.

Only two operators ship in Step-1: reverse_kl (topk_pathwise, design D1) and confidence
(unchanged BCE, moved into this framework). k3 / reinforce are deferred (D2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from verl.trainer.ppo.core_algos import agg_loss  # pure aggregation (masked_mean); no PPO here (D3)

from recipe.dspark_opd.loss_bridge import (
    block_decay_weight_mask as _decay_weight_mask,
    confidence_accept_rate_topk,
    flatten_blocks_to_sequence,
)

DSparkLossFn = Callable[["DSparkLossContext", object], tuple[torch.Tensor, dict[str, torch.Tensor]]]

DSPARK_LOSS_REGISTRY: dict[str, DSparkLossFn] = {}


def register_dspark_loss(name: str) -> Callable[[DSparkLossFn], DSparkLossFn]:
    """Register a DSpark loss operator under `name` (matches DSparkOPDLossConfig terms keys)."""

    def deco(fn: DSparkLossFn) -> DSparkLossFn:
        if name in DSPARK_LOSS_REGISTRY:
            raise ValueError(f"DSpark loss operator {name!r} already registered.")
        DSPARK_LOSS_REGISTRY[name] = fn
        return fn

    return deco


def get_dspark_loss_fn(name: str) -> DSparkLossFn:
    if name not in DSPARK_LOSS_REGISTRY:
        raise ValueError(f"Unknown DSpark loss operator {name!r}; registered: {list(DSPARK_LOSS_REGISTRY)}.")
    return DSPARK_LOSS_REGISTRY[name]


@dataclass
class DSparkLossContext:
    """Per-micro-batch inputs shared by all operators (all on device; 4D block tensors).

    S_grad:          [B,A,blk,K] student logπ WITH grad, on the K candidate ids. Cache path: gathered
                     on the FIXED S2 top-k ids. T5 path: draft's OWN top-K (only_stu) true full-vocab
                     logπ (loss_bridge.draft_topk_logp).
    T_on_S:          [B,A,blk,K] teacher logπ (no-grad), INDEX-ALIGNED to S_grad (same K ids). Cache:
                     teacher gathered on S2 ids. T5: teacher top-64 aligned to draft's K ids with
                     min-fill (loss_bridge.align_teacher_to_draft).
    S_logp_old:      [B,A,blk,K] no-grad old student logπ (rollout, confidence target) or None
                     (T5 path — confidence target derived from S_grad/T_on_S instead).
    eval_mask:       [B,A,blk] bool valid-token mask.
    decay_mask:      [B,A,blk] float = eval_mask · exp(-pos/γ); the ONE place decay is applied.
    confidence_pred: [B,A,blk] confidence-head logit WITH grad, or None if no head.
    block_size:      int (blk).
    """

    S_grad: torch.Tensor
    T_on_S: torch.Tensor
    S_logp_old: Optional[torch.Tensor]
    eval_mask: torch.Tensor
    decay_mask: torch.Tensor
    confidence_pred: Optional[torch.Tensor]
    block_size: int
    # Draft-OPD DUAL STREAM (T5 path). accept_mask/reject_mask [B,A,blk] bool split eval_mask into
    # the accept slots (draft proposals accepted) and the reject/boundary slot (target correction).
    # None on the cache path -> reject_kl operator no-ops and reverse_kl aggregates over eval_mask
    # as before (single-stream, unchanged). decay_mask already folds eval_mask·exp(-pos/γ); the
    # per-stream operators re-derive their own decay-weighted masks from accept/reject + the same γ.
    accept_mask: Optional[torch.Tensor] = None
    reject_mask: Optional[torch.Tensor] = None
    loss_decay_gamma: Optional[float] = None
    # forward KL (T5 accept stream, 2026-07-20). Support = TEACHER top-K (same K as reverse):
    #   S_on_T  [B,A,blk,K] draft logπ on the teacher top-K ids (grad, logsumexp).
    #   T_topk  [B,A,blk,K] teacher top-K logπ (no-grad; = t_logp[..., :K]).
    # None -> forward_kl operator no-ops (cache path / forward KL disabled).
    S_on_T: Optional[torch.Tensor] = None
    T_topk: Optional[torch.Tensor] = None


@register_dspark_loss("reverse_kl")
def reverse_kl_loss_op(ctx: DSparkLossContext, cfg) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Top-K reverse-KL surrogate, pathwise/analytic gradient (design D1 topk_pathwise).

        p_θ(j) = softmax_K(S_grad)_j                    # grad, normalized over the K candidates
        L(b,a,k) = Σ_j p_θ(j) · ( S_grad_j − T_on_S_j ) # = E_{p_θ}[ logπ_s − logπ_t ]  on top-k

    This is the differentiable, analytic-gradient form of the SAME weighted reverse-KL objective
    the old REINFORCE path optimized (its reward was rm_j = w_j·(T_j − S_j), w_j = softmax_K(S));
    here the gradient flows analytically through BOTH the softmax weights and logπ (low variance)
    instead of via a score-function estimator. Note S_grad/T_on_S are FULL-vocab log-probs gathered
    on the top-k ids (not top-k-renormalized), so this is a top-k importance-weighted estimate of
    reverse KL and is NOT guaranteed non-negative per token — that is expected and matches the
    original reward decomposition. T_on_S is detached (teacher frozen). reward_weight_mode is
    unused here — the weight IS softmax_K(S_grad), part of the objective and differentiable.
    """
    if cfg.reverse_kl_mode != "topk_pathwise":
        raise NotImplementedError(f"reverse_kl_mode={cfg.reverse_kl_mode!r} not implemented (D2).")
    per_token = _reverse_kl_per_token(ctx)                  # [B,A,blk] shared accept+reject math
    with torch.no_grad():
        m = ctx.eval_mask.bool()
        valid = per_token[m]
        mean = valid.mean() if valid.numel() else per_token.new_tensor(0.0)
    return per_token, {"actor/reverse_kl_raw": mean.detach()}


def _reverse_kl_per_token(ctx: DSparkLossContext) -> torch.Tensor:
    """Shared top-K reverse-KL per-token surrogate (used by both accept & reject streams).
    L(b,a,k) = Σ_j softmax_K(S)_j · (S_j − T_j). S grad, T detached (teacher frozen)."""
    S = ctx.S_grad.float()                                  # [B,A,blk,K] grad
    T = ctx.T_on_S.float().detach()                         # [B,A,blk,K] no-grad
    p = torch.softmax(S, dim=-1)                            # [B,A,blk,K]
    return (p * (S - T)).sum(dim=-1)                        # [B,A,blk]


@register_dspark_loss("reject_kl")
def reject_kl_loss_op(ctx: DSparkLossContext, cfg) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Draft-OPD DUAL STREAM: SAME top-K reverse-KL as `reverse_kl`, but this term is aggregated
    over the REJECT stream only (compose masks it with reject_mask). The per-token math is identical
    (one draft forward covers both streams); the split is purely in the aggregation mask + weight,
    so the reject/correction positions can carry a different loss weight than the accept positions.
    No-ops (per-token zero) when reject_mask is None (cache path — single stream)."""
    if cfg.reverse_kl_mode != "topk_pathwise":
        raise NotImplementedError(f"reverse_kl_mode={cfg.reverse_kl_mode!r} not implemented (D2).")
    if ctx.reject_mask is None:
        return torch.zeros_like(ctx.decay_mask), {}
    per_token = _reverse_kl_per_token(ctx)                  # [B,A,blk]
    with torch.no_grad():
        m = ctx.reject_mask.bool()
        valid = per_token[m]
        mean = valid.mean() if valid.numel() else per_token.new_tensor(0.0)
    return per_token, {"actor/reject_kl_raw": mean.detach()}


@register_dspark_loss("forward_kl")
def forward_kl_loss_op(ctx: DSparkLossContext, cfg) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Top-K FORWARD KL surrogate, mass-covering (design §4b). Aggregated over the ACCEPT stream
    (compose masks it with accept_mask).

        p_t(j) = softmax_K(T_topk)_j                    # teacher prob on teacher top-K (no-grad)
        L(b,a,k) = Σ_j p_t(j) · ( T_topk_j − S_on_T_j ) # = E_{p_t}[ logπ_t − logπ_s ] on teacher top-K

    Support/weight = TEACHER top-K (symmetric to reverse_kl's draft top-K / softmax_K(S)). Gradient
    flows only through S_on_T (draft logπ on the teacher ids); teacher terms are constant (frozen).
    Not guaranteed per-token non-negative (full-vocab logπ gathered on top-K, not re-normalized) —
    same caveat as reverse_kl. No-ops when S_on_T/T_topk are None (forward KL disabled / cache path).
    """
    if ctx.S_on_T is None or ctx.T_topk is None:
        return torch.zeros_like(ctx.decay_mask), {}
    S = ctx.S_on_T.float()                                  # [B,A,blk,K] grad (draft on teacher ids)
    T = ctx.T_topk.float().detach()                         # [B,A,blk,K] no-grad (teacher top-K)
    p_t = torch.softmax(T, dim=-1)                          # [B,A,blk,K] teacher prob on top-K
    per_token = (p_t * (T - S)).sum(dim=-1)                 # [B,A,blk]
    with torch.no_grad():
        m = (ctx.accept_mask if ctx.accept_mask is not None else ctx.eval_mask).bool()
        valid = per_token[m]
        mean = valid.mean() if valid.numel() else per_token.new_tensor(0.0)
    return per_token, {"actor/forward_kl_raw": mean.detach()}


@register_dspark_loss("confidence")
def confidence_loss_op(ctx: DSparkLossContext, cfg) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Acceptance-rate BCE (DSpark-specific; unchanged math, moved into the operator framework).

    target = Σ_j min(p_draft_j, p_target_j) on the top-k support (detached), predicted by the
    confidence head logit. Per-token BCE (not pre-multiplied by decay; compose weights it).
    """
    device = ctx.decay_mask.device
    if ctx.confidence_pred is None:
        # no confidence head -> zero per-token loss (compose skips a zero term cleanly).
        return torch.zeros_like(ctx.decay_mask), {}
    # confidence target = Σ_k min(p_draft, p_target) on the top-k support (detached).
    # cache path: use S_logp_old (no-grad rollout logp) vs T_on_S. T5 path (S_logp_old is None):
    # derive from S_grad/T_on_S (already the draft top-K + aligned teacher; detach S — target no-grad).
    S_for_conf = ctx.S_logp_old if ctx.S_logp_old is not None else ctx.S_grad.detach()
    accept = confidence_accept_rate_topk(S_for_conf, ctx.T_on_S)           # [B,A,blk] detached ∈[0,1]
    per_token = F.binary_cross_entropy_with_logits(
        ctx.confidence_pred.float(), accept.detach(), reduction="none")  # [B,A,blk]
    with torch.no_grad():
        m = ctx.eval_mask.bool()
        acc_mean = accept[m].mean() if m.any() else accept.new_tensor(0.0)
    return per_token, {"actor/confidence_accept_rate": acc_mean.to(device).detach()}


def compose_dspark_loss(ctx: DSparkLossContext, cfg) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum of enabled operators' decay-weighted token-mean.

    For each (name, weight) with weight != 0:
        per_token = op(ctx, cfg)                                  # [B,A,blk], no decay
        term = agg_loss(flatten(per_token), flatten(decay_mask), loss_agg_mode)  # decay applied ONCE
    Returns (total_loss, metrics). decay_mask carries eval_mask · exp(-pos/γ) so the decay is
    applied exactly once (num and den both weighted) — fixing the old PG path's decay-squared
    artifact and matching SFT's single decay.
    """
    # Per-term aggregation mask (Draft-OPD dual stream). Each term aggregates over its OWN stream:
    #   reverse_kl / forward_kl -> accept stream   reject_kl -> reject stream   confidence -> full eval.
    # On the cache path accept_mask/reject_mask are None, so reverse_kl falls back to the full
    # eval_mask (single-stream, byte-identical to before) and reject_kl/forward_kl are disabled by weight.
    gamma = ctx.loss_decay_gamma
    full_decay = ctx.decay_mask                                            # eval_mask · exp(-pos/γ)

    def _decay_for(name: str) -> torch.Tensor:
        if name in ("reverse_kl", "forward_kl") and ctx.accept_mask is not None:
            base = ctx.accept_mask                                         # accept stream
        elif name == "reject_kl" and ctx.reject_mask is not None:
            base = ctx.reject_mask                                         # reject stream
        else:
            return full_decay                                             # confidence / cache path
        return _decay_weight_mask(base, ctx.block_size, gamma)

    total: Optional[torch.Tensor] = None
    metrics: dict[str, torch.Tensor] = {}
    for name, weight in cfg.enabled_terms():
        per_token, term_metrics = get_dspark_loss_fn(name)(ctx, cfg)   # [B,A,blk]
        if cfg.loss_max_clamp is not None:
            c = float(cfg.loss_max_clamp)
            per_token = per_token.clamp(min=-c, max=c)
        mask_flat = flatten_blocks_to_sequence(_decay_for(name).unsqueeze(-1)).squeeze(-1)  # [B,A*blk]
        term_loss = agg_loss(
            loss_mat=flatten_blocks_to_sequence(per_token),
            loss_mask=mask_flat,
            loss_agg_mode=cfg.loss_agg_mode,
        )
        total = term_loss * weight if total is None else total + term_loss * weight
        metrics[f"actor/{name}_loss"] = term_loss.detach()
        metrics.update(term_metrics)
    assert total is not None, "compose_dspark_loss: no enabled terms (config validation should prevent this)."
    metrics["actor/loss"] = total.detach()
    return total, metrics


__all__ = [
    "DSPARK_LOSS_REGISTRY",
    "DSparkLossContext",
    "register_dspark_loss",
    "get_dspark_loss_fn",
    "compose_dspark_loss",
]
