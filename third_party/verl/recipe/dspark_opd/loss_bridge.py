"""DSpark-OPD loss bridge (S4): reward + 4D↔3D flatten + confidence — pure functions.

No verl deps beyond torch, so scripts/opd/s4_smoke.py can test the S4 math in isolation.
The GRAD training forward itself is NOT here: it goes through Qwen3DSparkModel.forward's OPD
branch (module(...) / FSDP.forward) so FSDP registers cross-rank gradient hooks — see
recipe/dspark_opd/worker.py:update_dspark_opd and docs/DSpark-OPD.md §S4. (An earlier version
had a standalone dspark_block_train_forward that called model submethods directly, bypassing
FSDP.forward → multi-GPU gradients never synced; removed in favor of the forward branch.)

Pipeline (tensor-contract §S4):
  1. logp_on_topk_ids : grad student logπ on the FIXED S2 top-k candidate ids -> [B,A,blk,K],
     computed from module(...).draft_logits (already markov-corrected == rollout corrected).
  2. build_opd_reward : rm = w_j·(T_on_S − S_logp), w_j=softmax_K(S_logp)  (only_stu/student_p),
     all no-grad (reward is a constant advantage for token_reward_direct).
  3. flatten_blocks_to_sequence / unflatten_sequence_to_blocks : [B,A,blk,*] ↔ [B,A*blk,*]
     (A*blk collapses into verl's response_len; MUST be invertible or reward misaligns).
  4. confidence target (top-K approx, LOCKED): accept_rate = Σ_k min(p_draft_k, p_target_k)
     on the student top-k support (= 1 − ½·TV restricted to top-k); detached BCE target.
"""
from __future__ import annotations

import torch


def logp_on_topk_ids(corrected_logits, top_k_ids, temperature=1.0):
    """GRAD student logπ on the FIXED top-k candidate ids. -> [B,A,blk,K].

    corrected_logits [B,A,blk,V] (grad, = module(...).draft_logits, markov-corrected),
    top_k_ids [B,A,blk,K] (fixed from S2). Uses the SAME temperature-scaled log_softmax as
    rollout (block_rollout.py logp_all) so on-policy old_log_prob (S2) and this recomputed
    log_prob match at ratio≈1.
    """
    temp = float(temperature) if float(temperature) > 0 else 1.0
    logp_all = torch.log_softmax(corrected_logits.float() / temp, dim=-1)     # [B,A,blk,V]
    return logp_all.gather(-1, top_k_ids)                                    # [B,A,blk,K]


@torch.no_grad()
def build_opd_reward(logp_target_on_topk, student_top_k_logp, *, weight_mode="student_p"):
    """OPD token reward on top-k (only_stu). rm = w_j·(T_on_S − S_logp). -> [B,A,blk,K].

    Mirrors Rethink-OPD dp_actor.compute_distillation_reward only_stu branch:
      rm_scores = -kl_val * w,  kl_val = S_logp - T_on_S  =>  rm = w·(T_on_S − S_logp).
    w_j = softmax_K(S_logp) for student_p (teacher_p uses T_on_S; none uses uniform).
    No-grad: the reward is a fixed advantage for token_reward_direct (advantage=const).
    """
    S = student_top_k_logp.float()
    T = logp_target_on_topk.float()
    if weight_mode == "student_p":
        w = torch.softmax(S, dim=-1)
    elif weight_mode == "teacher_p":
        w = torch.softmax(T, dim=-1)
    elif weight_mode == "none":
        w = torch.full_like(S, 1.0 / S.shape[-1])
    else:
        raise ValueError(f"unknown weight_mode: {weight_mode}")
    rm = w * (T - S)                                                         # [B,A,blk,K]
    return rm


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


def confidence_bce(confidence_pred, accept_rate, weight_mask):
    """Weighted BCE-with-logits: confidence_pred (grad logits) vs detached accept_rate.

    Mirrors deepspec/modeling/dspark/loss.py confidence branch (num/den form).
    Returns scalar (num/den) or 0 if no confidence head. weight_mask [B,A,blk] float.
    """
    if confidence_pred is None:
        return torch.zeros((), device=accept_rate.device)
    errs = torch.nn.functional.binary_cross_entropy_with_logits(
        confidence_pred.float(), accept_rate.detach(), reduction="none") * weight_mask
    den = weight_mask.sum().clamp_min(1.0)
    return errs.sum() / den


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


def unflatten_sequence_to_blocks(x, A, blk):
    """[B, A*blk, *] -> [B, A, blk, *]. Inverse of flatten_blocks_to_sequence."""
    B = x.shape[0]
    return x.reshape(B, A, blk, *x.shape[2:])
