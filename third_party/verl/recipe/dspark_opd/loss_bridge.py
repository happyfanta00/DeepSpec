"""DSpark-OPD loss bridge — pure candidate/prob helpers for the dual-stream KL loss.

No verl deps beyond torch. The GRAD training forward itself is NOT here (it goes through
Qwen3DSparkModel.forward's OPD branch so FSDP registers cross-rank grad hooks). These are the
pure functions the fused train_step + losses.py compose over draft_logits / teacher top-K:
  - logp_on_ids_lse   : draft logπ on a GIVEN id set (forward KL — teacher top-K), logsumexp.
  - draft_topk_logp   : draft's OWN top-K ids + logπ (reverse KL support), logsumexp.
  - align_teacher_to_draft : teacher logπ aligned to draft's top-K ids (min-fill outside teacher top-Kt).
  - confidence_accept_rate_topk : Σ min(p_draft,p_target) on top-K (detached BCE target).
  - block_decay_weight_mask / flatten_blocks_to_sequence : decay mask + [B,A,blk,*]->[B,A*blk,*].
"""
from __future__ import annotations

import torch


def logp_on_ids_lse(corrected_logits, ids, temperature=1.0):
    """GRAD student logπ on a GIVEN id set, via logsumexp (no full-vocab log_softmax materialized).

    corrected_logits [B,A,blk,V] (grad, = module(...).draft_logits). ids [B,A,blk,K] long (any id
    set — here the TEACHER top-K, for forward KL). logπ = logit_gathered/temp − logsumexp(V), the
    TRUE full-vocab log-prob (same caliber as draft_topk_logp / reverse-KL's S_grad), but gathering
    only the K ids instead of building the [B,A,blk,V] log_softmax tensor (forward KL support is the
    teacher's ids, not the argmax, so we gather rather than topk). Returns [B,A,blk,K] (grad).
    """
    temp = float(temperature) if float(temperature) > 0 else 1.0
    z = corrected_logits.float() / temp
    lse = torch.logsumexp(z, dim=-1, keepdim=True)                           # [B,A,blk,1] grad
    z_on = z.gather(-1, ids)                                                 # [B,A,blk,K] grad
    return z_on - lse                                                        # true full-vocab logπ (grad)


def draft_topk_logp(corrected_logits, k, temperature=1.0):
    """T5 candidate step: draft's OWN top-k (ids + TRUE full-vocab logπ, WITH grad).

    corrected_logits [B,A,blk,V] (grad). Returns (ids [B,A,blk,k] long no-grad,
    logp [B,A,blk,k] grad). logπ = logit/temp − logsumexp(logit/temp) computed via logsumexp
    (scalar reduction over V, no [B,A,blk,V] log_softmax tensor materialized). This is the
    student `only_stu` candidate set for the T5 loss (K=log_prob_top_k, e.g. 16).
    """
    temp = float(temperature) if float(temperature) > 0 else 1.0
    z = corrected_logits.float() / temp
    kk = min(int(k), z.shape[-1])
    lse = torch.logsumexp(z, dim=-1, keepdim=True)                           # [B,A,blk,1] grad
    top_logits, ids = torch.topk(z, kk, dim=-1)                              # [B,A,blk,kk]
    logp = top_logits - lse                                                  # true full-vocab logπ (grad)
    return ids, logp


def align_teacher_to_draft(draft_ids, teacher_ids, teacher_logp):
    """T5 candidate algo: teacher logπ on the DRAFT's top-k ids, filled from the teacher's own
    top-`kt` (ids/logp). For each draft id: if it is in the teacher's top-`kt`, use the teacher's
    true logπ there; else use min(teacher top-`kt` logπ) as a lower-bound substitute (an id outside
    the teacher's top-`kt` has teacher logπ ≤ that min).

    draft_ids     [B,A,blk,K]  long  (student top-K candidate ids)
    teacher_ids   [B,A,blk,kt] long  (teacher top-kt ids)
    teacher_logp  [B,A,blk,kt] float (teacher top-kt true logπ)
    -> T_on_S [B,A,blk,K] float (no grad; teacher frozen).
    """
    # min over teacher's top-kt (its smallest logπ) — the substitute for out-of-top-kt ids.
    t_min = teacher_logp.min(dim=-1, keepdim=True).values                    # [B,A,blk,1]
    # match each draft id against every teacher id: eq[...,i,j] = (draft_i == teacher_j)
    eq = draft_ids.unsqueeze(-1) == teacher_ids.unsqueeze(-2)                 # [B,A,blk,K,kt]
    hit = eq.any(dim=-1)                                                      # [B,A,blk,K] in teacher top-kt?
    # gathered teacher logp for matched ids (argmax picks the matching j; 0 when no hit — masked below)
    j = eq.float().argmax(dim=-1)                                            # [B,A,blk,K]
    gathered = teacher_logp.gather(-1, j)                                    # [B,A,blk,K]
    return torch.where(hit, gathered, t_min.expand_as(gathered))             # [B,A,blk,K]


@torch.no_grad()
def confidence_accept_rate_topk(student_top_k_logp, logp_target_on_topk):
    """Top-K-support accept-rate target (LOCKED approx). accept_rate = Σ_k min(p_d, p_t).

    accept_rate = 1 − ½·Σ_V|p_d − p_t| = Σ_V min(p_d, p_t); restricted to the student top-k
    support (both logp are on the SAME S2 top-k ids). Detached BCE target. -> [B,A,blk].
    """
    p_d = student_top_k_logp.float().exp()
    p_t = logp_target_on_topk.float().exp()
    accept = torch.minimum(p_d, p_t).sum(dim=-1)                            # [B,A,blk]
    return accept.clamp_(0.0, 1.0)


def block_decay_weight_mask(eval_mask, block_size, loss_decay_gamma):
    """eval_mask × exp(−pos/γ) block-internal decay (loss.py:_build_loss_weight_mask). -> [B,A,blk]."""
    w = eval_mask.to(torch.float32)
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        pos = torch.arange(block_size, device=eval_mask.device).view(1, 1, -1)
        w = w * torch.exp(-pos.float() / float(loss_decay_gamma))
    return w


def flatten_blocks_to_sequence(x):
    """[B, A, blk, *] -> [B, A*blk, *] (A,blk collapse into verl response_len). Row-major."""
    B, A, blk = x.shape[:3]
    return x.reshape(B, A * blk, *x.shape[3:])
