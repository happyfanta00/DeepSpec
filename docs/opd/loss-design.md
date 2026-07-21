# DSpark-OPD Loss 设计（数学 + 代码）

> 本文档说明 DSpark On-Policy Distillation（OPD）训练的 loss 是如何设计的，分别从**数学角度**与**代码实现角度**给出。
>
> 相关代码：
> - 纯数学函数：`third_party/verl/recipe/dspark_opd/loss_bridge.py`（无 verl 依赖，`scripts/opd/s4_smoke.py` 可孤立测）
> - 装配 + backward：`third_party/verl/recipe/dspark_opd/worker.py`（`DSparkActorRolloutRefWorker.train_step` / `update_dspark_opd`）
> - 复用 verl core：`third_party/verl/verl/trainer/ppo/core_algos.py`（`token_reward_direct` + 3D `compute_policy_loss_vanilla`）
> - 设计背景：`docs/DSpark-OPD.md §2.5`

---

## 总览

DSpark-OPD 的总损失是两项之和：

```
L = L_PG  +  α · L_conf
     └ 反向KL策略梯度   └ 接受率 BCE（DSpark 专有）
```

- `α` = `confidence_head_alpha`（默认 `1.0`）。
- **没有独立的蒸馏 loss**——蒸馏信号全部藏在 reward / advantage 里，复用 verl 的 PPO 策略梯度管线（`token_reward_direct` + 3D top-k policy loss）。
- Student = 草稿模型；Teacher = target 模型（Qwen3-4B，自蒸馏）。

---

## 一、数学角度

### 符号约定

- 对一条 response，随机采 `A=32` 个 anchor，每个 anchor 处草稿采样 `blk=7` 个 token（rollout）。
- 在每个 block 位置取草稿 **top-K 候选**（`K=16`，`only_stu` 策略）。
- 索引 `(b, a, k)`：样本 `b`、anchor `a`、块内位置 `k`；候选 `j ∈ {1..K}`。
- `S_j = logπ_student(y_j)`、`T_j = logπ_target(y_j)`，均在**同一批 S2 top-k 候选 id** 上取。

### (1) Token 级 reward = 加权反向 KL 的 top-k 分解

```
rm_j = w_j · (T_j − S_j),      w_j = softmax_K(S)_j      (student_p)
```

- 本质是 `−KL(student ‖ teacher)` 在 top-K 候选集上的加权分解：teacher 比 student 更看好某 token（`T_j > S_j`）→ reward 为正 → 推动草稿提高该 token 概率。
- **全程 no-grad**：reward 是一个常量 advantage（`token_reward_direct` 的输入）。
- `T_j` 来自路线 A 的 online 块对角打分（条件于草稿采样前缀）；`S_j` 是 S2 rollout 顺手取的 no-grad 旧 log-prob。

### (2) Advantage = `token_reward_direct`

```
A_j = rm_j · m
```

直接把 reward 当 advantage，**不做任何 GAE / GRPO 组内归一化**。`m` 是权重掩码（见下），3D 时广播到 `(B, T, 1)`。

### (3) 策略梯度 loss：dual-clip PPO（3D，对 K 求和）

```
r_j = exp( logπ_θ(y_j) − logπ_old(y_j) )
ℓ_j = max( −A_j·r_j , −A_j·clip(r_j, 1−ε_lo, 1+ε_hi) )    (A<0 再加 dual-clip 下界 c=3.0)
L_PG = token-mean_(b,a,k)( Σ_{j=1..K} ℓ_j )
```

**on-policy 关键退化**：代码里 `old_log_prob = log_prob.detach()`，故 `r_j ≡ 1`，裁剪失效，退化为

```
L_PG ≈ − Σ_(b,a,k) m_(b,a,k) · Σ_{j=1..K} A_j · logπ_θ(y_j)
```

即「用 teacher–student KL 差作权重的 top-K 加权策略梯度」——这正是 OPD 的目标函数。

**权重掩码**（block 内位置衰减，沿用 SFT 的 `loss_decay_gamma`）：

```
m_(b,a,k) = eval_mask_(b,a,k) · exp(−k / γ),      γ = 4.0
```

`eval_mask` 充当 verl 的 `response_mask`，屏蔽无效块 / padding。

### (4) Confidence BCE（DSpark 专有，verl 无对应）

接受率目标用 **top-K 支撑近似**（全词表 TV 距离在 top-k 上的限制）：

```
accept_rate_(b,a,k) = Σ_{j=1..K} min(p_draft_j, p_target_j)  ≈  1 − ½·TV(draft, target)
```

```
                Σ_(b,a,k) m_(b,a,k) · BCE-with-logits( ĉ_(b,a,k), sg[accept_rate_(b,a,k)] )
L_conf = ────────────────────────────────────────────────────────────────────────────────
                                    Σ_(b,a,k) m_(b,a,k)
```

`ĉ` 是 confidence 头输出的 logit（带梯度），target 被 detach（`sg[·]`），是加权 BCE 的 num/den 形式。

---

## 二、代码实现角度

### 分层结构

| 层 | 位置 | 职责 |
|---|---|---|
| 纯数学函数 | `loss_bridge.py` | reward / 展平 / confidence / decay，无 verl 依赖 |
| 装配 + backward | `worker.py` `train_step` / `update_dspark_opd` | 前向 → reward → loss → backward → step |
| 复用 verl core | `core_algos.py` | `token_reward_direct`、3D `compute_policy_loss_vanilla` |

> `train_step` 是当前默认（融合 rollout→teacher→update 三步为一次 worker RPC）；`update_dspark_opd` 是 S4 遗留的独立 update RPC，loss 逻辑相同。

### 每个 micro-batch 内的六步（`worker.py`）

```python
# 1) 带梯度前向（必须经 module(...) = FSDP.forward，否则多卡梯度不同步）
block_prev = cat([anchor_tok, ỹ_1..ỹ_{blk-1}])   # teacher-force markov 头到采样 token
out = module(input_ids, target_hidden_states, loss_mask,
             anchor_positions=固定, block_keep_mask, block_prev_tokens=采样)
S_grad = logp_on_topk_ids(out.draft_logits, top_k_ids)   # 在固定 top-k id 上带梯度 gather

# 2) reward + advantage（no-grad）
rm    = build_opd_reward(T_on_S, S_logp_old, weight_mode="student_p")   # w·(T−S)
dmask = block_decay_weight_mask(eval_mask, blk, γ)                      # eval_mask·exp(-k/γ)
adv   = token_reward_direct(flatten(rm), flatten(dmask))                # A = rm·mask

# 3) 3D dual-clip PG（on-policy: old = log_prob.detach() → ratio≈1）
pg_loss = compute_policy_loss_vanilla(
    old_log_prob=flatten(S_grad).detach(), log_prob=flatten(S_grad),
    advantages=adv, response_mask=flatten(dmask), loss_agg_mode="token-mean")

# 4) confidence BCE
accept    = confidence_accept_rate_topk(S_logp_old, T_on_S)   # Σ min(p_d,p_t)
conf_loss = confidence_bce(out.confidence_pred, accept, dmask)
loss = pg_loss + conf_alpha * conf_loss

# 5) 梯度累积
(loss * scale).backward()          # scale = 1/n_micro

# 6)（micro 循环结束后）
grad_norm = self.actor._optimizer_step()   # grad-clip + optimizer.step
```

### 数学 ↔ 代码 对照表

| 数学 | 代码 | 位置 |
|---|---|---|
| `w_j·(T_j−S_j)`，no-grad | `build_opd_reward` | `loss_bridge.py:38-58` |
| `A = rm·m`，3D 广播 | `token_reward_direct` | `core_algos.py:854-880` |
| 4D block→3D 序列展平 `[B,A,blk,*]→[B,A·blk,*]` | `flatten_blocks_to_sequence` | `loss_bridge.py:97-100` |
| dual-clip、对 K 求和 | `pg_losses = torch.sum(..., dim=-1)` | `core_algos.py:1144` |
| on-policy 退化 `r≈1` | `old_log_prob = logp_flat.detach()` | `worker.py:332` |
| `m = eval_mask·exp(-k/γ)` | `block_decay_weight_mask` | `loss_bridge.py:88-94` |
| `Σ min(p_d,p_t)` | `confidence_accept_rate_topk` | `loss_bridge.py:61-71` |
| 加权 BCE num/den | `confidence_bce` | `loss_bridge.py:74-85` |
| 梯度累积 `1/n_micro` | `(loss*scale).backward()` | `worker.py:537` |

### 两个易踩的正确性约束（代码里刻意处理）

1. **梯度同步双层修复**：训练前向必须走 `module(...)`（`FSDP.forward`，注册跨 rank all-reduce hook），且 actor 用 `NO_SHARD` 而非 `fsdp_size=1` 的退化 HYBRID mesh——缺任一层多卡梯度都不同步、副本静默发散（详见 `docs/DSpark-OPD.md §2.6.5` 及 memory `dspark-opd-multigpu-grad-sync-bug`）。
2. **top-k 候选同源**：`S_logp_old`（reward 里的旧 logp）与 `S_grad`（loss 里的带梯度 logp）都在**同一批 S2 固定的 `student_top_k_ids`** 上 gather，保证 advantage 与 log-prob 的候选集严格对应（apples-to-apples），且都基于 markov 校正后分布，使 `ratio≈1`。

---

## 一句话总结

DSpark-OPD 的 loss = **top-K 加权反向 KL 策略梯度**（`w_j·(T_j−S_j)` 当 reward → `token_reward_direct` advantage → 3D dual-clip PG，on-policy 下退化为 `−Σ m·A·logπ_θ`）**+ 接受率 BCE**（target = top-k 上 `Σ min(p_d,p_t)`），block 位置带 `exp(-k/γ)` 衰减，micro-batch `1/n_micro` 累积后一次 optimizer step。前者复用 verl 现成 3D 管线，后者是 DSpark 为投机解码接受率自加的头。
