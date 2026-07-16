#!/usr/bin/env python3
"""Stage-M1 核心正确性验证：坐实 upstream sglang 的原生 DSPARK rejection sampler
与本 repo evaluator 的 rejection sampling 是【同一个采样器】（完整-draft-分布，即
消费 draft 分布 q，接受概率 min(1,p/q)，拒绝按残差 (p-q)+ 重采样）。

对拍两条腿：
  A. upstream: sglang.srt.speculative.reject_sampling.chain_speculative_sampling_triton
     （Triton 内核，DSPARK verify 实际调用的那个；coin*q<p + relu(p-q)）。
  B. repo:     deepspec.utils.sampling 的 rejection sampling
     （base_evaluator.py:252-285 同款：accept_prob=min(1,p/q)，拒绝 sample_residual）。

两种验证，互补：
  (1) 分布等价（蒙特卡洛）：对固定的 (p, q, 候选链)，各跑 N 次，比较
      「最终 commit 的 token」经验分布 + 接受长度分布。等价 ⇒ 大样本下一致。
      更强的是：理论上二者对每个 slot 的接受概率都是 min(1,p/q)，输出边缘分布都等于 p。
  (2) 目标分布无损（数学必然）：无论 draft q 如何，最终每步输出的 token 边缘分布
      应精确等于 target p。这对两条腿都必须成立（这是投机解码无损的定义）。

用法：
    python scripts/opd/dspark_rejection_equivalence_check.py [--trials 200000] [--vocab 32] [--gamma 4]
需要：CUDA（Triton 内核）。在 upstream env (~/.venv/dspark-opd-sglang) 下跑。
"""
from __future__ import annotations

import argparse
import sys

import torch


def repo_chain_rejection(candidates, target_probs, draft_probs, coins, coin_final):
    """Reference chain rejection sampling matching base_evaluator.py:252-285 +
    deepspec.utils.sampling.sample_residual, done per-batch on the SAME inputs
    the Triton kernel gets. Returns (num_accept[bs], final_token[bs]).

    candidates:   [bs, S]      draft token chain (candidates[:,0] is the root anchor,
                               candidates[:,1:] are the S-1 proposed draft tokens)
    target_probs: [bs, S, V]   p at each slot (row i is the target dist AFTER
                               consuming candidates[:, :i])
    draft_probs:  [bs, S-1, V] q for each proposed draft token slot (slot k aligns
                               with candidates[:, k+1]); no draft dist for the bonus.
    coins:        [bs, S-1]    uniforms for accept test at each proposed slot
    coin_final:   [bs]         uniform for final residual/target sampling
    """
    bs, S, V = target_probs.shape
    num_steps = S - 1  # number of proposed draft tokens (== gamma)
    num_accept = torch.zeros(bs, dtype=torch.int64, device=target_probs.device)
    final_token = torch.empty(bs, dtype=torch.int64, device=target_probs.device)

    for b in range(bs):
        cur_row = 0  # which target_probs row is "current" (after accepted prefix)
        accepted = 0
        rejected = False
        for step in range(1, num_steps + 1):
            tok = int(candidates[b, step].item())
            p = float(target_probs[b, cur_row, tok].item())
            q = float(draft_probs[b, step - 1, tok].item())
            coin = float(coins[b, step - 1].item())
            # accept iff coin < p/q  <=>  coin*q < p  (kernel form). min(1,p/q) rule.
            if coin * q < p:
                accepted += 1
                cur_row = step
            else:
                rejected = True
                break
        num_accept[b] = accepted
        # final sampling
        p_row = target_probs[b, cur_row]                 # [V]
        if not rejected:
            # all proposed accepted -> bonus from pure target p
            dist = p_row
        else:
            q_row = draft_probs[b, cur_row]              # cur_row <= num_steps-1 on reject
            dist = torch.clamp(p_row - q_row, min=0.0)
            s = dist.sum()
            if float(s.item()) <= 1e-8:
                dist = p_row
                s = dist.sum()
            dist = dist / s.clamp_min(1e-8)
        # inverse-CDF sample with coin_final (matches kernel's cumulative pass)
        cdf = torch.cumsum(dist, dim=-1)
        u = float(coin_final[b].item()) * float(dist.sum().item())
        idx = int(torch.searchsorted(cdf, torch.tensor(u, device=cdf.device)).item())
        idx = min(idx, V - 1)
        final_token[b] = idx
    return num_accept, final_token


def upstream_chain_rejection(candidates, target_probs, draft_probs, coins, coin_final):
    """Call the exact Triton kernel DSPARK verify uses."""
    from sglang.srt.speculative.reject_sampling import chain_speculative_sampling_triton

    bs, S, V = target_probs.shape
    device = target_probs.device
    # chain buffers: retrive_index = arange (single linear chain), next/sibling unused here
    retrive_index = torch.arange(bs * S, dtype=torch.int64, device=device).view(bs, S)
    row_next = torch.arange(1, S + 1, dtype=torch.int64, device=device)
    row_next[-1] = -1
    retrive_next_token = row_next.unsqueeze(0).expand(bs, -1).clone()
    retrive_next_sibling = torch.full((bs, S), -1, dtype=torch.int64, device=device)
    predicts = torch.full((bs * S,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, S), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros((bs,), dtype=torch.int32, device=device)

    chain_speculative_sampling_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coin_final,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    num_accept = accept_token_num.to(torch.int64)
    # final token lives at predicts[last_accepted_global_idx]; last accepted retrive idx
    # is retrive_index[b, num_accept[b]] (root when 0 accepted).
    rows = torch.arange(bs, device=device)
    last_idx = retrive_index[rows, num_accept]
    final_token = predicts[last_idx].to(torch.int64)
    return num_accept, final_token


def make_case(bs, S, V, device, seed_offset):
    """Random p (target), q (draft), and a candidate chain sampled from q."""
    g = torch.Generator(device=device)
    g.manual_seed(1234 + seed_offset)
    target_probs = torch.softmax(
        torch.randn(bs, S, V, generator=g, device=device) * 1.5, dim=-1
    ).float()
    draft_probs = torch.softmax(
        torch.randn(bs, S - 1, V, generator=g, device=device) * 1.5, dim=-1
    ).float()
    # candidate chain: root token arbitrary (slot 0 not tested), proposed tokens ~ q
    candidates = torch.empty(bs, S, dtype=torch.int64, device=device)
    candidates[:, 0] = torch.randint(0, V, (bs,), generator=g, device=device)
    for k in range(S - 1):
        candidates[:, k + 1] = torch.multinomial(
            draft_probs[:, k], num_samples=1, generator=g
        ).squeeze(-1)
    return candidates, target_probs, draft_probs, g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200_000, help="MC trials per case")
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--gamma", type=int, default=4, help="proposed draft tokens (S = gamma+1)")
    ap.add_argument("--tol", type=float, default=0.01, help="max abs prob diff tolerance")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for the Triton kernel."); return 2
    device = torch.device("cuda")
    V, S = args.vocab, args.gamma + 1
    N = args.trials

    print(f"[cfg] trials={N} vocab={V} gamma={args.gamma} (S={S}) tol={args.tol}")

    def first_commit(num_accept, final_token, candidates):
        # first committed token: candidates[:,1] if >=1 accepted, else the
        # residual/target sample at the current row.
        return torch.where(num_accept >= 1, candidates[:, 1], final_token)

    def emp_dist(x, k):
        return torch.bincount(x, minlength=k).float() / x.numel()

    # -------------------------------------------------------------------------
    # TEST 1 — A vs B EQUIVALENCE: fixed (p,q,chain), SAME coins fed to both
    # samplers. Identical inputs ⇒ if the two are the same sampler, outputs match.
    # -------------------------------------------------------------------------
    candidates1, tp1, dp1, g = make_case(1, S, V, device, seed_offset=0)
    candidates = candidates1.expand(N, S).contiguous()
    target_probs = tp1.expand(N, S, V).contiguous()
    draft_probs = dp1.expand(N, S - 1, V).contiguous()
    coins = torch.rand(N, S - 1, generator=g, device=device, dtype=torch.float32)
    coin_final = torch.rand(N, generator=g, device=device, dtype=torch.float32)

    up_acc, up_tok = upstream_chain_rejection(candidates, target_probs, draft_probs, coins, coin_final)
    M = min(N, 20_000)  # repo python loop is O(N*S); cap for speed
    rp_acc, rp_tok = repo_chain_rejection(
        candidates[:M], target_probs[:M], draft_probs[:M], coins[:M], coin_final[:M]
    )
    up_fc = first_commit(up_acc, up_tok, candidates)
    rp_fc = first_commit(rp_acc, rp_tok, candidates[:M])

    tok_ab = float((emp_dist(up_fc, V) - emp_dist(rp_fc, V)).abs().max().item())
    acc_ab = float((emp_dist(up_acc, S) - emp_dist(rp_acc[:M], S)).abs().max().item())
    print(f"\n[TEST 1: A vs B equivalence, identical inputs]")
    print(f"  max |Δ first-commit-token dist| = {tok_ab:.4f}  (M={M})")
    print(f"  max |Δ accept-len dist|         = {acc_ab:.4f}")
    print(f"  mean accept len: upstream={up_acc.float().mean():.4f} repo={rp_acc[:M].float().mean():.4f}")

    # -------------------------------------------------------------------------
    # TEST 2 — TARGET LOSSLESSNESS: resample the candidate chain from q EACH trial
    # (marginalize over the draft), + resample coins. The first-committed-token
    # marginal must equal target p0 for BOTH samplers. This is the definition of
    # a lossless speculative sampler (draft q washes out; output ~ target p).
    # -------------------------------------------------------------------------
    p0 = tp1[0, 0]  # [V] fixed target next-token dist
    # per-trial candidates ~ q
    cand2 = torch.empty(N, S, dtype=torch.int64, device=device)
    cand2[:, 0] = torch.randint(0, V, (N,), generator=g, device=device)
    for k in range(S - 1):
        cand2[:, k + 1] = torch.multinomial(dp1[:, k].expand(N, V), 1, generator=g).squeeze(-1)
    tp2 = tp1.expand(N, S, V).contiguous()
    dp2 = dp1.expand(N, S - 1, V).contiguous()
    coins2 = torch.rand(N, S - 1, generator=g, device=device, dtype=torch.float32)
    coinf2 = torch.rand(N, generator=g, device=device, dtype=torch.float32)

    up_acc2, up_tok2 = upstream_chain_rejection(cand2, tp2, dp2, coins2, coinf2)
    up_fc2 = first_commit(up_acc2, up_tok2, cand2)
    up_vs_p0 = float((emp_dist(up_fc2, V) - p0).abs().max().item())

    rp_acc2, rp_tok2 = repo_chain_rejection(cand2[:M], tp2[:M], dp2[:M], coins2[:M], coinf2[:M])
    rp_fc2 = first_commit(rp_acc2, rp_tok2, cand2[:M])
    rp_vs_p0 = float((emp_dist(rp_fc2, V) - p0).abs().max().item())

    print(f"\n[TEST 2: target losslessness, candidates resampled from q]")
    print(f"  max |upstream first-commit dist - target p0| = {up_vs_p0:.4f}  (N={N})")
    print(f"  max |repo     first-commit dist - target p0| = {rp_vs_p0:.4f}  (M={M})")

    # MC tolerance scales ~1/sqrt(n); use looser bound for the M-sized repo runs.
    tol_N = args.tol
    tol_M = max(args.tol, 3.0 / (M ** 0.5))
    ok = (tok_ab < args.tol) and (acc_ab < args.tol) \
        and (up_vs_p0 < tol_N) and (rp_vs_p0 < tol_M)
    print(f"\n[tol] equivalence<{args.tol} lossless_N<{tol_N:.4f} lossless_M<{tol_M:.4f}")
    print("RESULT:", "EQUIVALENT & LOSSLESS ✅" if ok else "MISMATCH ❌ (raise --trials if borderline)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
