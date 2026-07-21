# DSpark-OPD Loss 解耦重构设计（Step 1：反向KL + Confidence 自定义算子）

> 状态：**已实现并验证**（s4_smoke + loss_ops_smoke 全绿；见 §10）
> 目标读者：DSpark-OPD 维护者
>
> **决策摘要**：D1=`topk_pathwise`（top-K 反向 KL 解析梯度，唯一反向 KL 算子）；
> D2=暂不实现 `reinforce`/`k3` 可切算子；D3=保留 `core_algos.agg_loss` 依赖；
> D4=归一化口径先不改（沿用现状 token-mean + `1/n_micro` + FSDP 平均梯度）。
> 关联文档：`docs/opd/loss-design.md`（现状）、`docs/DSpark-OPD.md §2.5`（loss 映射）、
> `/home/ec2-user/efs_data/workspace/Draft-OPD/docs/loss_design_analysis.md`（参考工程方案）

---

## 0. 目标与非目标

**目标（Step 1）**：把 DSpark-OPD 的 loss 计算与 verl 的「reward → advantage → PPO 策略梯度」管线**彻底解耦**。反向 KL loss 与 confidence loss 完全由**自定义 loss 算子**直接实现并反向传播，不再经过 `token_reward_direct` + `compute_policy_loss_vanilla`。同时把 loss 架构改造成**算子注册表 + 配置驱动**，为未来扩展更多 loss 形式（forward KL、top-k 全分布 KL、rejected-draft 流等）留出干净的接口。

**非目标（本次不做）**：
- 不引入新的 loss 语义（forward KL、rejected-draft 流等留待后续；本次只把现有「反向 KL + confidence」平移到新框架）。
- 不改 rollout / teacher 打分 / 数据契约（`block_rollout.py` / `teacher_scoring.py` / tensor-contract 不动）。
- 不改 FSDP / 梯度同步 / micro-batch 累积机制（NO_SHARD、`module(...)` 前向、`1/n_micro` 累积保持不变）。

---

## 1. 现状分析：耦合点在哪里

现状（`docs/opd/loss-design.md`）总损失：

```
L = L_PG + α·L_conf
     └ 反向KL「策略梯度」   └ 接受率 BCE
```

`L_PG` 的计算路径（`worker.py` `train_step` Phase-3，`update_dspark_opd` 同构）：

```python
rm  = build_opd_reward(T_on_S, S_logp_old, weight_mode)      # rm_j = w_j·(T_j − S_j), no-grad
adv = token_reward_direct(flatten(rm), flatten(decay_mask))  # A = rm·mask (verl adv estimator)
pg_loss = compute_policy_loss_vanilla(                       # 3D dual-clip PPO
    old_log_prob=lp.detach(), log_prob=lp, advantages=adv,
    response_mask=flatten(decay_mask), loss_agg_mode="token-mean", config=cfg_pl)
```

**耦合的三个 verl 组件**：
| 组件 | 位置 | 现在承担的职责 |
|---|---|---|
| `build_opd_reward` | `loss_bridge.py:38` | 把反向 KL 拆成 top-k 加权 reward `w_j·(T_j−S_j)` |
| `token_reward_direct` | `core_algos.py:854` | reward 直接当 advantage，mask 广播 |
| `compute_policy_loss_vanilla`（3D 分支）| `core_algos.py:1058/1118` | dual-clip PPO，对 K 求和，token-mean |

### 1.1 现状本质：score-function（REINFORCE）估计子，非「反向 KL 的解析梯度」

on-policy 下 `old_log_prob = log_prob.detach()` → `ratio ≡ 1`，PPO 裁剪整段是**死代码**，退化为

```
∇L_PG ≈ − Σ_(b,a,k) mask · Σ_j  sg[w_j·(T_j − S_j^old)] · ∇ logπ_θ(y_j)
```

即把 `−KL_topk` 当常量 reward、梯度只经**采样 token 的 logπ**回传的 **REINFORCE / score-function 估计子**。它是反向 KL 目标的一个**高方差、有偏（top-k 截断）**的梯度估计，而不是反向 KL 的直接可微 loss。这正是解耦要改掉的东西——参考方案（Draft-OPD）的 `use_policy_gradient=False` 分支就是把 KL 当**监督式可微 loss 直接反传**（`losses.py:692`）。

### 1.2 现状的两个副作用（解耦会顺带修掉）

1. **位置衰减被平方（PG 与 confidence 衰减不一致）**。`token_reward_direct` 已把 `decay_mask` 乘进 advantage（`adv = rm·mask`），`compute_policy_loss_vanilla` 又把同一个 `decay_mask` 当 `response_mask` 做 `token-mean`。`masked_mean(pg_losses, mask) = Σ(pg·mask)/Σmask`，而 `pg` 里已含一层 mask →**分子含 `mask²`、分母含 `mask¹`**。`eval_mask` 是 0/1 不受影响，但 `exp(−pos/γ)` 被平方成 `exp(−2·pos/γ)`——**PG 项的有效 γ 只有配置值的一半**。而 confidence 项（`confidence_bce`）只乘一次 `decay_mask`，衰减正确。**两项衰减口径当前不一致**，与 SFT（`loss.py:_build_loss_weight_mask` 单次衰减）也不一致。

2. **一堆无意义的配置与 metric**。`cfg_pl`（`clip_ratio*`、`clip_ratio_c`）、`algorithm.adv_estimator=token_reward_direct`、metric `actor/ppo_kl`、`pg_clipfrac` 在 `ratio≡1` 下全无意义，纯属为了迁就 PPO 接口。

### 1.3 已经解耦的部分：confidence loss

`confidence_bce`（`loss_bridge.py:74`）本就是**直接可微 BCE**（`confidence_pred` logit vs detached `accept_rate`），与 PPO 无关。本次只需把它**并入新算子框架**，数学不变。

---

## 2. 参考方案要点（Draft-OPD）与对 DSpark 的取舍

Draft-OPD（`verl/trainer/distillation/losses.py`）值得借鉴的**工程骨架**：

1. **算子注册表 + 装饰器**：`register_distillation_loss(DistillationLossSettings(names=[...]))` → `DISTILLATION_LOSS_REGISTRY`，`get_distillation_loss_fn(loss_mode)` 按名取算子（`losses.py:80/96`）。新增一种 loss 只是「写一个函数 + 挂一个装饰器」。
2. **统一算子签名**：每个算子 `(config, model_output, data) -> (per_token_losses, metrics)`，返回**逐 token loss 张量**（不是标量），聚合与归一化在**外层统一**做（`distillation_loss` 编排，`losses.py:555`）。
3. **可微 KL 估计子做纯函数**：反向 KL 用 `kl_penalty(..., "k3")`（`core_algos.py`，`k3 = exp(t−s)−(t−s)−1`，恒非负、低方差），forward KL 用 `_local_bernoulli_forward_kl`（`losses.py:421`）。这些是**只依赖 logp 的可微函数**，直接反传。
4. **`use_policy_gradient` 是配置开关**：同一套 per-token loss，既能直接反传（监督），也能当 advantage 走 PG（`losses.py:671/692`）。

**对 DSpark 的取舍**：
- ✅ 采纳「注册表 + 统一算子签名 + per-token loss + 外层聚合」。这是「未来扩展更多 loss」的核心。
- ⚠️ **不照搬 Draft-OPD 的单采样 KL**。Draft-OPD 的 `kl_penalty` 作用在**单个采样 token 的标量 logp** 上；DSpark 走的是 **top-K 稠密**（`K=16`, `student_top_k_ids`）。DSpark 的反向 KL 算子应是 **top-K 上的反向 KL**（见 §4），把现有 `w_j = softmax_K(S)` 的加权自然纳入。
- ❌ 不引入 Draft-OPD 的 `no_padding_2_padding` / nested-tensor / SP 切分那套（DSpark 是 `[B,A,blk,K]` 4D 块结构 + NO_SHARD，不适用）。DSpark 的算子直接吃 4D 块张量。

---

## 3. 目标架构

### 3.1 分层（沿用现有三层，只替换「装配」层的中段）

| 层 | 位置 | 变化 |
|---|---|---|
| 纯数学算子 + 注册表 | **新增 `recipe/dspark_opd/losses.py`** | 反向KL / confidence 算子 + `compose_dspark_loss` 编排 |
| loss 配置 | **新增 `recipe/dspark_opd/loss_config.py`** | `DSparkOPDLossConfig` dataclass（算子清单 + 权重 + 归一化） |
| 装配 + backward | 改 `worker.py`（`train_step` / `update_dspark_opd`） | Phase-3 中段：删 reward/adv/ppo，改调 `compose_dspark_loss` |
| 复用底层 | `loss_bridge.py` | 保留纯 helper（`logp_on_topk_ids`/`flatten_*`/`block_decay_weight_mask`/`confidence_accept_rate_topk`）；`build_opd_reward` 降级为「reinforce 兼容算子」专用 |

**不再依赖** `core_algos.token_reward_direct` / `compute_policy_loss_vanilla` / `get_adv_estimator_fn`。

### 3.2 算子注册表与统一契约

```python
# recipe/dspark_opd/losses.py
DSPARK_LOSS_REGISTRY: dict[str, DSparkLossFn] = {}

def register_dspark_loss(name):                     # 装饰器，仿 Draft-OPD
    def deco(fn): DSPARK_LOSS_REGISTRY[name] = fn; return fn
    return deco

@dataclass
class DSparkLossContext:
    """一个 micro-batch 前向后、所有算子共享的输入（4D 块张量）。"""
    S_grad:      torch.Tensor   # [B,A,blk,K] 带梯度 student top-k logπ (logp_on_topk_ids)
    T_on_S:      torch.Tensor   # [B,A,blk,K] teacher top-k logπ（同一批 top_k_ids）
    S_logp_old:  torch.Tensor   # [B,A,blk,K] no-grad 旧 logπ（rollout）
    eval_mask:   torch.Tensor   # [B,A,blk] bool
    decay_mask:  torch.Tensor   # [B,A,blk] float = eval_mask · exp(-pos/γ)（衰减只在此处产生一次）
    confidence_pred: Optional[torch.Tensor]  # [B,A,blk] logit，带梯度
    block_size:  int

# 契约：算子返回「逐 token loss [B,A,blk]」+ metrics（不做聚合、不预乘 decay）
DSparkLossFn = Callable[[DSparkLossContext, DSparkOPDLossConfig],
                        tuple[torch.Tensor, dict[str, torch.Tensor]]]
```

### 3.3 编排：统一的 decay-加权 token-mean（修掉 §1.2 的衰减平方）

```python
def compose_dspark_loss(ctx, cfg):
    total = None; metrics = {}
    for name, weight in cfg.enabled_terms():          # 例如 [("reverse_kl",1.0), ("confidence",α)]
        if weight == 0: continue
        per_token, m = DSPARK_LOSS_REGISTRY[name](ctx, cfg)   # [B,A,blk]，未乘 decay
        # decay-加权 token-mean：Σ(loss·decay_mask)/Σ(decay_mask)（decay 只在此乘一次，num/den 一致）
        term_loss = agg_loss(                              # 复用 verl agg_loss（纯聚合，无 PPO）
            loss_mat=flatten_blocks_to_sequence(per_token),
            loss_mask=flatten_blocks_to_sequence(ctx.decay_mask.unsqueeze(-1)).squeeze(-1),
            loss_agg_mode=cfg.loss_agg_mode)              # 默认 "token-mean"
        total = term_loss*weight if total is None else total + term_loss*weight
        metrics[f"actor/{name}_loss"] = term_loss.detach()
    metrics["actor/loss"] = total.detach()
    return total, metrics
```

- `agg_loss` 是 verl 的**纯聚合函数**（`core_algos.py:942`，`masked_mean`），不含 PPO 语义，保留它避免重造轮子。也可内联为 DSpark 自己的 `masked_mean` 彻底断开 core_algos 依赖（见 §8 决策 D3）。
- **衰减只在 `decay_mask` 里产生一次**，num/den 同乘，反向 KL 与 confidence 口径一致，且与 SFT 一致 → 修掉 §1.2.1。

---

## 4. 反向 KL loss 算子（决策：top-K 解析梯度）

现有 reward 是「反向 KL 在 top-K 上的加权分解」`rm_j = w_j·(T_j − S_j)`, `w_j = softmax_K(S)`。解耦后把它写成**可微 loss**。**已定 D1 = top-K 反向 KL 的解析（pathwise）梯度**，作为**唯一**的反向 KL 算子实现：

### `topk_pathwise`（已定，唯一实现）

```
p_θ(j) = softmax_K(S_grad)_j                    # 带梯度，K 个候选上归一
L_revkl(b,a,k) = Σ_j p_θ(j) · ( S_grad_j − T_on_S_j )     # E_{p_θ}[logπ_s − logπ_t]，top-k 上
```

- 这是现有**加权反向 KL 目标**在 top-K 支撑上的可微（解析梯度）实现。梯度同时经 `softmax` 权重与 `logπ`，**低方差**，是「自定义 loss 算子」最自然的形态。
- 与现状的联系：现状 REINFORCE 的 reward 是 `rm_j = w_j·(T_j−S_j)`, `w_j=softmax_K(S_old)`，梯度 `−Σ_j sg[rm_j/w_j·...]·∇S_j`；本式梯度 `∇L = Σ_j[∇p_θ·(S_j−T_j) + p_θ·∇S_j]`。**优化同一目标，但用解析梯度取代 score-function 估计**——正是解耦的意义。
- 数值实现：`p_θ = softmax(S_grad, dim=-1)`，`L = Σ_j p_θ·(S_grad − T_on_S.detach())`。
- **注意不保证逐 token 非负**：`S_grad`/`T_on_S` 是在 top-k id 上 gather 的**全词表 log-prob**（非 top-k 重归一），故 `L` 是反向 KL 的 top-k 重要性加权估计，逐 token 可正可负——这与原 reward 分解一致、符合预期（`S==T` 时为 0）。可选 `loss_max_clamp` 双向截断。
- 稳定性：`T_on_S`、`S_grad` 都在同一批 top-k id 上（同源，`docs/opd/loss-design.md §正确性约束2`），差值有界。
- 注意 `reward_weight_mode`（`student_p`/`teacher_p`/`none`）在此算子下**不再需要**——权重就是 `p_θ = softmax_K(S_grad)`，是 KL 的一部分、且带梯度。配置里 `reward_weight_mode` 仅为旧路径遗留，本算子忽略。

### 未实现的备选（决策 D2：暂不做）

以下两种作为可切算子**本次不实现**，仅记录以备后续消融需要时再加（届时只是往注册表加一个函数）：
- **`k3` 单采样估计子**（`kl_penalty(..., "k3") = exp(t−s)−(t−s)−1`）：贴 Draft-OPD，但丢弃 top-K 稠密信号，退化为单采样、方差更高。
- **`reinforce` 兼容算子**（`−Σ_j sg[w_j·(T_j−S_j^old)]·S_grad_j`，复用 `build_opd_reward`）：与现状 PG 梯度同向，本可作回归对照。**因 D2 不注册为算子**；但其对拍价值在 §6.3 用「临时内联对照」的方式保留（不进产品代码）。

---

## 4b. 前向 KL loss 算子（新增，2026-07-20；accept 流改用）

**动机**：反向 KL（`topk_pathwise`）在 **draft top-K** 支撑上、权重 `p_θ=softmax_K(S)`（draft 侧），是「mode-seeking」——draft 只需在自己已高概率的 token 上贴近 teacher。改用**前向 KL** `KL(p_teacher ‖ p_draft)`，权重是 **teacher 概率**、支撑是 **teacher top-K**，是「mean-seeking / mass-covering」——强制 draft 覆盖 teacher 认为重要的 token。**本次决策：accept 流（draft 提议被接受的位）改用前向 KL；reject 流保持反向 KL 不变**（见 §双流）。

> **K 统一**：forward / reverse 用**同一个 K = `log_prob_top_k`（当前 16）**——只是候选来源不同（forward 用 teacher top-K，reverse 用 draft top-K）。K 由 config 单一控制，改 K 时两方向一起变。（前提 `K ≤ Kt=dspark_teacher_top_k=64`，取 teacher top-64 的前 K 个即 teacher top-K。）

### `forward_kl`（新算子）

支撑集 = **teacher top-K**（复用合并前向已产的 teacher top-`Kt` `t_ids/t_logp` 的**前 K 个**——`topk` 已按 logπ 降序，切片即得，无需新前向）。

```
# 记 teacher top-K 的 id = t_ids[..., :K]，其 full-vocab logπ = t_logpK = t_logp[..., :K]（no-grad，teacher 冻结）
p_t(j)      = softmax_K( t_logpK )_j                   # teacher 概率，在 K 个候选内重归一（决策：top-K 内 softmax）
S_on_T_j    = draft 在 t_idsK[j] 上的 full-vocab logπ   # 带梯度，logsumexp 方式（不物化 [.,V] log_softmax）
L_fwdkl(b,a,k) = Σ_j p_t(j) · ( t_logpK_j − S_on_T_j )  # E_{p_t}[ logπ_t − logπ_s ]，teacher top-K 上
```

- **权重 `p_t = softmax_K(t_logpK)`（no-grad）**：在 teacher top-K 内重归一（与反向 KL 的 `p_θ=softmax_K(S)` 完全对称——同一个 K，那边 draft 侧 top-K 内 softmax，这边 teacher 侧 top-K 内 softmax）。teacher 冻结，权重不带梯度。
- **梯度只经 `S_on_T`（draft logπ）**：`∇L = −Σ_j p_t(j)·∇S_on_T_j`。这是标准 forward-KL 的可微形式（teacher 项是常量），梯度把 draft 概率往 teacher top-K 的每个 token 上推、按 teacher 权重加权。**低方差**（解析梯度，非采样）。
- **draft logπ 取法（决策：logsumexp）**：`S_on_T = draft_logits.gather(t_idsK) / temp − logsumexp(draft_logits/temp, V)`，标量 logsumexp 归约，不物化 `[micro,A,blk,V]` 的 log_softmax（A=256 时省显存）。是**真 full-vocab logπ**，与反向 KL 的 `S_grad` 口径一致。
- **逐 token 非负性**：`t_logpK` 与 `S_on_T` 都是 full-vocab logπ（非 top-K 重归一），故 `L` 是前向 KL 的 top-K 重要性加权估计，逐 token 可正可负（`p_t` 归一但 `logπ_t−logπ_s` 差值未必同号）——与反向 KL 的口径说明一致，符合预期（`S==T` 时为 0）。可选 `loss_max_clamp` 截断。
- **与反向 KL 的对称性**：reverse `Σ softmax_K(S)·(S−T)`（支撑 draft top-K、权重 draft）；forward `Σ softmax_K(T)·(T−S)`（支撑 teacher top-K、权重 teacher）。**同一个 K**，两者共用 `compose` 的 decay-加权 token-mean 编排。

### 与合并前向的对接（无需新前向）

- teacher 侧：合并前向已产 `t_ids/t_logp [B,A,blk,Kt]`；forward KL 取 `[..., :K]`。
- draft 侧：draft 前向已产 `draft_logits [micro,A,blk,V]`（grad）；在 teacher top-K id 上 gather logπ（logsumexp）→ 新量 `S_on_T [micro,A,blk,K]`（grad）。这是 **draft 前向后新增的一次 gather**（现有只 gather 了 draft 自己的 top-K `d_ids/S_grad`）。

### 双流

- **accept 流**：`compose` 对 `forward_kl` 用 `accept_mask` 聚合（替代原 accept 流的 `reverse_kl`）。
- **reject 流**：`reject_kl`（反向 KL on draft top-K）**保持不变**（本次只改 accept 流）。
- 三个流的权重（config）：`dspark_loss_forward_kl_weight`（accept 流，新）、`dspark_loss_reject_kl_weight`（reject 流）、`dspark_loss_confidence_weight`。原 `dspark_loss_reverse_kl_weight` 若同时 >0，则 accept 流是 forward+reverse 叠加（默认把它设 0，accept 流纯 forward KL）。

---

## 5. Confidence loss 算子（数学不变，仅并框架）

```python
@register_dspark_loss("confidence")
def confidence_loss_op(ctx, cfg):
    if ctx.confidence_pred is None:
        return zeros[B,A,blk], {}
    accept = confidence_accept_rate_topk(ctx.S_logp_old, ctx.T_on_S)   # Σ_j min(p_d,p_t), detached
    per_token = F.binary_cross_entropy_with_logits(
        ctx.confidence_pred.float(), accept.detach(), reduction="none")  # [B,A,blk]，未乘 decay
    return per_token, {"actor/confidence_accept_rate": ...}
```
- 与现 `confidence_bce`（`loss_bridge.py:74`）等价：现状 `Σ(bce·decay)/Σ(decay)` 就是 `compose` 的 decay-加权 token-mean，**归一化统一到编排层**后数值不变。
- `accept_rate` target 仍用 no-grad 的 `S_logp_old`（apples-to-apples，与现状一致）。

---

## 6. 代码变动范围

### 6.1 新增
- **`recipe/dspark_opd/losses.py`**：注册表 + `DSparkLossContext` + `reverse_kl`（**仅 `topk_pathwise` 一个实现**，D2）+ `confidence` 算子 + `compose_dspark_loss`。注册表接口预留（后续加 `k3`/`reinforce` 只是加函数），但本次不注册它们。
- **`recipe/dspark_opd/loss_config.py`**：`DSparkOPDLossConfig` dataclass（见 §7），含校验（权重非负、`terms` 键必须已注册）。`reverse_kl_mode` 字段可保留但当前只接受 `topk_pathwise`（其它值报错，提示「未实现」）。

### 6.2 修改
- **`recipe/dspark_opd/worker.py`**（两处，`train_step` Phase-3 + `update_dspark_opd`，逻辑同构）：
  - 删除：`build_opd_reward` + `get_adv_estimator_fn("token_reward_direct")` + `compute_policy_loss_vanilla` + `cfg_pl`（clip ratios）。
  - 新增：构造 `DSparkLossContext`（`S_grad`/`T_on_S`/`S_logp_old`/`eval_mask`/`decay_mask`/`confidence_pred`），调 `compose_dspark_loss(ctx, loss_cfg)` 得 `loss, metrics`。
  - `loss_cfg` 在 `init_model` 时从 `override_config.dspark_loss` 构造一次，存 `self._loss_cfg`。
  - metrics：`actor/pg_loss`/`actor/ppo_kl` → `actor/reverse_kl_loss`/`actor/confidence_loss`/`actor/loss`（保留 `grad_norm`/`n_micro`）。
  - `import` 清理：移除 `core_algos` 的 PPO import。
- **`recipe/dspark_opd/trainer.py:77-79`**：打印的 metric key 改名（`pg_loss`→`reverse_kl_loss`，去掉 `ppo_kl`）。
- **`recipe/dspark_opd/config/dspark_trainer.yaml`**：
  - `override_config` 下新增 `dspark_loss:` 块（§7），移除旧 `reward_weight_mode`/`loss_decay_gamma`/`confidence_head_alpha` 平铺键。
  - `algorithm.adv_estimator: token_reward_direct` 保留但加注释说明其被解耦 loss 旁路（仅为满足 verl schema）。
- **`loss_bridge.py`**：`build_opd_reward` 保留（`s4_smoke` 的 `reinforce` 内联参照仍用它）；其余 helper（`logp_on_topk_ids`/`flatten_*`/`block_decay_weight_mask`/`confidence_accept_rate_topk`）原样复用。`confidence_bce` 保留供 `s4_smoke` 对拍。
- **`docs/opd/loss-design.md`**：改写为解耦后的数学/代码（本次未改，留待后续；本设计文档 §10 已记录落地实现）。

### 6.3 测试（实际落地见 §10）
- **`scripts/opd/s4_smoke.py`**（真实模型，GPU）检查 C/D/E/F：
  1. **C** `topk_pathwise` 反向 KL 正确性：与手算 `Σ_j p_θ·(S−T)` 对齐、有限、`requires_grad`；`S==T` 时 loss=0（**不断言逐 token ≥0 / 梯度=0**——见 §10 订正1）。
  2. **D** 接线对拍（临时内联，不进产品代码）：内联 `reinforce` 参照 `−Σ_j sg[rm_j]·S_grad`（复用 `build_opd_reward`），与旧 PG 路径在**同一单次 `decay_mask`** 下对比梯度（Δ=0）；并对比旧生产路径的**双重衰减**梯度差，量化衰减平方修复。**该参照仅存在于 smoke，不注册为算子**（D2）。
  3. **E** `confidence` 算子 vs 旧 `confidence_bce` → `allclose`；compose backward 后可训练参数有梯度、冻结头无梯度。
  4. **F** `1/n_micro` 累积 == whole-batch：**同一前向图**上对输出叶子求梯度（隔离 bf16 批前向噪声，见 §10 订正2）。
- **`scripts/opd/loss_ops_smoke.py`**（纯 torch，CPU）：config 解析/校验 + 三算子 + compose 的可孤立回归测。

---

## 7. 配置变更

在 `override_config` 下新增**扁平标量键**（⚠️ 不能用嵌套 dict）：

```yaml
override_config:
  ...
  # ⚠️ 必须是扁平标量键。verl 的 update_model_config（utils/model.py:66）会把
  # override_config 里 dict 类型的值当作嵌套 HF sub-config 递归 setattr →
  # 嵌套 `dspark_loss:` 块会在 Qwen3Config 上抛 AttributeError。故用扁平 `dspark_loss_*`。
  dspark_loss_reverse_kl_weight: 1.0    # reverse_kl 权重
  dspark_loss_confidence_weight: 1.0    # confidence 权重（原 confidence_head_alpha）
  dspark_loss_reverse_kl_mode: topk_pathwise  # 当前仅支持 topk_pathwise（k3/reinforce 未实现，D2）
  dspark_loss_decay_gamma: 4.0          # block 内 exp(-pos/γ)，与 SFT 一致
  dspark_loss_agg_mode: token-mean      # 编排层聚合
  dspark_loss_max_clamp: null           # 可选，逐 token loss 双向截断（null=关）
```

`DSparkOPDLossConfig.from_override_config` 读这些扁平键并校验（`terms` 键必须已注册；`reverse_kl_mode != topk_pathwise` 报「未实现」）。向后兼容：无 `dspark_loss_*` 键时回落到旧默认（`reverse_kl=1`，`confidence=confidence_head_alpha`；`loss_decay_gamma` 亦可读旧键）。`reward_weight_mode` 在 `topk_pathwise` 下**不再使用**（权重即 `softmax_K(S_grad)`）。

---

## 8. 决策（已锁定）

- **D1｜反向 KL 算子形式 = `topk_pathwise`（已定）**：top-K 反向 KL 的解析（pathwise）梯度 `L = Σ_j softmax_K(S_grad)_j·(S_grad_j − T_on_S_j)`，作为唯一实现。**这是本次唯一影响训练数值的决策**——从 score-function 估计换到解析梯度。
- **D2｜暂不实现 `reinforce` / `k3` 可切算子（已定）**：注册表接口预留，本次只注册 `reverse_kl`(=topk_pathwise) 与 `confidence`。`reinforce` 的对拍价值改为 smoke 脚本内联参照（§6.3），不进产品代码。
- **D3｜保留 `core_algos.agg_loss` 依赖（已定）**：`compose` 直接用 verl 的纯聚合函数 `agg_loss`（`masked_mean`），不重造。后续若要零 core_algos 依赖再内联。
- **D4｜归一化口径先不改（已定）**：沿用现状「micro 内 token-mean + `1/n_micro` 累积 + FSDP 平均梯度」。SFT 式「全局 denom all-reduce × world_size」属正交问题，不混入本次解耦。

---

## 9. 可行性结论

**✅ 高可行**。解耦是「外科手术式」的：只替换 `worker.py` 两处 Phase-3 的**中段三行**（reward→adv→ppo）为一次 `compose_dspark_loss`，前向/backward/FSDP/micro 累积全不动；反向 KL 与 confidence 都是**只依赖已有 4D 张量**（`S_grad`/`T_on_S`/`S_logp_old`/`decay_mask`/`confidence_pred`）的纯函数。参考方案（Draft-OPD）已验证「注册表 + per-token 算子 + 外层聚合」的工程模式可行。附带收益：修掉衰减平方、清掉无意义的 PPO 配置/metric、为后续 loss 扩展留出注册接口。主要风险是 D1 带来的**训练数值变化**（从 score-function 换到解析梯度），用 §6.3 smoke 脚本里的 `reinforce` 内联参照可量化、可回退（回退即恢复 `worker.py` 旧三行）。

---

## 10. 实现记录（已落地）

**新增文件**：`recipe/dspark_opd/loss_config.py`（`DSparkOPDLossConfig`）、`recipe/dspark_opd/losses.py`（注册表 + `reverse_kl`/`confidence` 算子 + `compose_dspark_loss`）、`scripts/opd/loss_ops_smoke.py`（纯 torch 算子单测，CPU <1s）。
**修改文件**：`worker.py`（`init_model` 建 `self._loss_cfg`；`train_step` Phase-3 + `update_dspark_opd` 中段换成 `DSparkLossContext` + `compose_dspark_loss`；删 `OmegaConf`/PPO import/`cfg_pl`）、`trainer.py`（metric key）、`config/dspark_trainer.yaml`（`dspark_loss` 块 + `adv_estimator` 注释）、`scripts/opd/s4_smoke.py`（检查 C/D/E/F 重写）。

**验证**：`s4_smoke.py`（真实 4B draft+target，GPU）与 `loss_ops_smoke.py`（CPU）全绿。关键结论：
- **D（解耦精确性）**：新的单次衰减聚合 + `reinforce` 内联参照的梯度 == 旧 PG 路径单次衰减梯度，`max|Δ|=0`；旧生产路径的**双重衰减**梯度差 `0.128` —— 即本次修掉的衰减平方 artifact，被量化坐实。
- **C**：`reverse_kl` 算子 == 闭式 `Σ_j softmax_K(S)·(S−T)`（Δ=0），`S==T` 时 loss=0。
- **E**：`confidence` 算子 == 旧 `confidence_bce`（Δ=0），梯度只落可训练参数、冻结头无梯度。
- **F**：`1/n_micro` 累积 == whole-batch（同一前向图上对输出叶子求梯度，Δ=0）。

**实现期两处与初稿的订正**：
1. **反向 KL 非负性**：`topk_pathwise` 用的是在 top-k id 上 gather 的**全词表 log-prob**（非 top-k 重归一），故 `L` 是反向 KL 的 top-k 重要性加权估计，**逐 token 可正可负**（`S==T` 时为 0），与原 reward 分解一致。初稿误标「≥0 / = KL|_topk」，已在 §4 更正。
2. **F 测试的隔离**：跨两次 bf16 批前向比较**参数**梯度会混入 flex_attention 的批不结合噪声（与本重构无关）。改为在**同一前向图**上对输出叶子（`S_grad`/`confidence_pred`）比较梯度，干净地只测「`1/n_micro` 聚合」本身，Δ=0。
