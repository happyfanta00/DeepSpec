# DSpark-OPD 张量契约（dataflow tensor contract）

> 本文件定义 DSpark-OPD 单步 dataflow 中各阶段的**输入/输出张量**（名称、shape、dtype、语义）。
> 它是开发的**契约**：每实现/修改一个阶段（S1–S4），都应对照本文件核对张量的 shape/dtype/语义，
> 并在张量设计变化时**同步更新本文件**。主文档 `DSpark-OPD.md` §2.9 / 第 3 部分引用此文件。
>
> 状态标记：✅ 已实现并实测；🔷 已设计待实现。
>
> **⚠️ 架构说明（融合，最新）**：S2/S3/S4 的张量语义（rollout / teacher 打分 / reward+loss）**均不变、仍准确**——但运行时它们**不再是三次独立 worker RPC**，而是融合进**一个** `actor_rollout_wg.train_step` 内顺序执行（rollout→teacher→update，中间产物留 worker 显存，见 `DSpark-OPD.md` 架构演进 note + `fused-step-design.md`）。下文各阶段的"gate=xxx"、独立 `compute_rm_score`/`update_dspark_opd` RPC 描述属**历史开发记录**；底层纯函数（`dspark_block_rollout`/`score_blocks_flat`/`loss_bridge`）不变,融合内直接复用。

## 符号与常量（Qwen3-4B + `config/dspark/dspark_qwen3_4b.py`）

| 符号 | 含义 | 值 | 来源 |
|---|---|---|---|
| `B` | batch size（右填充后） | 如 2 | `data.train_batch_size` |
| `T` | padded 序列长（= batch 内 max real_len） | ≤ 4096 | `dspark_collate_fn` |
| `A` | num_anchors（每样本最多采 anchor 数） | **32**（OPD；SFT 用 512） | `model.num_anchors` |
| `blk` | block_size | 7 | `model.block_size` |
| `K` | **top-k 候选数（top-k 稠密 OPD，对齐 Rethink-OPD `LOG_PROB_TOP_K`）** | 16 | `override_config.log_prob_top_k` |
| `H` | hidden_size | 2560 | Qwen3-4B config |
| `L` | len(target_layer_ids) | 5 | `[1,9,17,25,33]` |
| `V` | vocab_size | 151936 | Qwen3-4B config |

> **注**：rollout 次数 `n`（= verl `rollout.n`，如 4）**吸收进 batch 维**（见下），故**不作为独立张量维**——各阶段张量 shape 里 `B` 已含 ×n。

> **OPD 信号双层多样性（已锁定，对齐 Rethink-OPD 默认）**：
> - **rollout 次数 `n`（序列/状态级多样性）**：每个缓存样本重复 `n=rollout.n=4` 次，各副本独立采自己的 anchor + block —— **吸收进 batch 维**（`B` → `B×n`），标准 verl 做法。因我们用 `token_reward_direct`（逐条独立、**无 GRPO 组内基线**），n 个副本**不需共享状态**，就是 n 份独立 on-policy 数据。
> - **top-k `K`（token/block 级多样性）**：每 block 位置取 top-`K` 候选算加权 KL（`only_stu` + `student_p`）。**默认 top-k 稠密**，`K=0` 退化为单采样 OPD。K 是 per-token 信号的**内维**，与 n 正交。
>
> **为何 R 进 batch 维（前提 + 理由）**：OPD 训练须**额外常驻 teacher 模型**，故 `num_anchors` 必然远小于 SFT 的 512（**设 A=32**）。A 小 → `B×n` 不再爆炸，可直接用 verl 标准 batch 机制。相比"R 内嵌样本"方案：① 张量去掉显式 R 维（简化）；② batch 归一化对齐 verl（`ppo_mini_batch_size *= rollout.n` `fsdp_workers.py:237` + `//= dp_size`）；③ block_rollout 变 R-agnostic（只处理拿到的 batch，各行独立采 anchor）。代价：n 个副本各自跑一次 backbone forward（A 小，相对 teacher 全词表 forward 是小头，可接受）。**⚠️ repeat 位置（优化#3 起改）**：`rollout.n` 的 repeat 现**默认在 worker 侧 `train_step`**（`repeat_interleave(n)`，driver 只 dispatch unique prompt），非 verl 标准的 driver 侧 `gen_batch.repeat`——见 `fused-step-design.md §优化#3`。`B×n` batch 语义不变，仅 repeat 发生地点变（`DSPARK_REPEAT_ON_DRIVER=1` 可复原 driver 侧）。

> **Rethink-OPD 默认 batch 口径（源码确认，我们对齐）**：`rollout.n=4`；`train_batch_size`（单位=prompt/样本）；`ppo_mini_batch_size`（输入单位=样本，verl 内部 `*= rollout.n` 归一到 response 粒度，`fsdp_workers.py:237`，再 `//= dp_size` 到单卡）；`ppo_micro_batch_size_per_gpu`（单位=response，单卡 micro batch）；梯度累积 = 单卡 mini / micro（`dp_actor.py:799`）。真实样本数 `= train_batch_size × n`（`utils/config.py:109`）。**我们完全走此标准链路**（不再有 §之前的 `rollout.n=1` hack）。

> 变长/padding：样本变长（真实长度 `real_len[b] ≤ T`），右填充到 `T`；padding 屏蔽靠 `loss_mask`（padding 处=0），**不靠 attention_mask**（见 `DSpark-OPD.md` S1 note）。

> **前向次数账（每训练 step，`only_stu` 默认）**：
> | | 学生(draft) 前向 | teacher 前向 | 总计 |
> |---|---|---|---|
> | **Rethink-OPD** | **3**：① vLLM 采样(no_grad) → ② `compute_log_prob` 补分布+old_logp+**top-k 候选**(no_grad) → ③ `update_policy` 训练前向(**带梯度**) | 1（`compute_rm_score`, no_grad） | **4** |
> | **DSpark（我们）** | **2**：① S2 `block_rollout` 采样(no_grad，**顺手出 top-k 候选 id**) → ② S4 训练前向(**带梯度**，算 loss) | 1（S3 块对角打分, no_grad） | **3** |
>
> - **DSpark 比 Rethink-OPD 省 1 次学生前向**：Rethink-OPD 的第②次(`compute_log_prob`)是为**弥补 vLLM 不保留 logits**（须重算分布出 top-k）；我们的 `block_rollout` 采样时已算出完整 `corrected_logits`，**top-k 候选 id 在 S2 顺手 `topk` 取**（no_grad，只取 id），故省掉这次独立前向。
> - **为何 S2(no_grad) 与 S4(带梯度) 不能合并成一次带梯度前向**：带梯度前向须保留计算图(全部激活)到 `backward`。若 S2 带梯度并保留到 S4，则学生计算图须在 **S3 teacher 完整前向期间一直驻留**显存 → 学生图 + teacher(4B) 图 + 两模型权重同时在显存 → OOM 风险。分两次是**用重算换显存**（把带梯度图的存活窗口压到最短、只在 S4 内），这是所有主流 PPO 框架的通用模式。
> - **on-policy 下 `old_log_prob` 无需独立前向**：`ppo_epochs=1`+单 mini-batch 时 verl 用 `log_prob.detach()` 当 old（`dp_actor.py:858`），ratio≡1；故第①/②次前向都不是为 old_logp。

> **各阶段 micro-batch 粒度对比（我们 vs Rethink-OPD）**：
> | 阶段 | Rethink-OPD | DSpark（我们） |
> |---|---|---|
> | rollout | vLLM 引擎自管连续批（`max_num_batched_tokens=32768`） | `dspark_block_rollout` **整卡 batch 一次**（no_grad，块并行 PyTorch，暂不切 micro） |
> | student 前向（补 top-k） | `compute_log_prob`，**dynamic_bsz** | **无此步**（rollout 顺手出 top-k，省一次前向） |
> | teacher 前向 | `compute_rm_score`，**静态** `micro_batch_size_per_gpu=24` | `score_blocks_flat` **整卡 batch 一次 batched forward**（no_grad；B 样本同一 forward，FULL_SHARD 只 all-gather 1 次；不切 micro） |
> | student 前向+后向 | `update_policy`，**dynamic_bsz**（`ppo_max_token_len_per_gpu=32768`，`loss_scale=样本数/mini`） | `update_dspark_opd`，**静态**切 micro + `1/n_micro`（照搬 verl 静态分支，见 `DSpark-OPD.md` §S4 note） |
>
> - **只有带梯度那步（`update_dspark_opd`）切了 micro**：它留激活、显存压力最大。rollout/teacher 是 no_grad 整卡一次——no_grad 不留激活，显存压力小。若 S6 多卡+大 `num_anchors` 下 rollout/teacher 也 OOM，再各自加独立 micro 循环（与 update 步的无关）。
> - **训练那步经 `FSDP.forward`（`module(...)`），非子方法直接调用**——这是多卡梯度同步的前提（见 §S4）；rollout/teacher 是 no_grad，前向经 `module(...)`（teacher）或子方法（rollout，`fsdp_size=1` 下参数全量故安全），无 backward 故不涉及梯度归约。
> - **静态 vs dynamic**：Rethink-OPD 用 dynamic_bsz 处理变长 response；我们块结构 per-sample 等长，静态 `1/n_micro` 与 dynamic `样本数/mini` 数值近似（`micro=1` 时严格相等），故维持静态（§S4 note 详述取舍）。

---

## 阶段 S1 — Dataset 输出（右填充 batch）✅

`DSparkCacheDataset.__getitem__` 逐样本（变长），`dspark_collate_fn` 右填充成 batch：

> **⚠️ hidden 来源随 `DSPARK_HIDDEN_MODE` 变（shape/contract 不变，只是来源变）**：默认 `recompute`
> 模式下 dataset **只出 `input_ids/loss_mask/attention_mask/position_ids`**（`adapt_tokens_only`），
> `target_hidden_states` 由 **worker 内 teacher 重算**（`recompute_target_hidden_states`）；仅
> `dispatch` 模式 dataset 出全量含 hidden，`cache` 模式 dataset 只出 `{sample_index}`、worker 读
> cache。下表的 `target_hidden_states`/`target_last_hidden_states` 行是**逻辑契约**（各模式最终喂给
> draft 的张量），非默认模式下不由 dataset 产出。见 `worker-side-cache-read-design.md`。

| 张量 | shape | dtype | 语义 |
|---|---|---|---|
| `input_ids` | `[B, T]` | long | 真实 token，右填充 0 |
| `loss_mask` | `[B, T]` | long | 1=监督位；padding 处=0 —— **padding 屏蔽的唯一依据** |
| `attention_mask` | `[B, T]` | long | 1=真实 token（供 verl gen 路径 pop；recompute 模式喂 teacher 重算 forward 屏蔽 padding；块 forward 不用） |
| `position_ids` | `[B, T]` | long | `arange(T)`（仅名义，同上） |
| `target_hidden_states` | `[B, T, L*H]` = `[B, T, 12800]` | bf16 | 5 中间层 hidden 拼接；draft cross-attn 的 context K/V 源（默认 recompute 重算 bf16；cache/dispatch 走 fp8→bf16 sanitize） |
| `target_last_hidden_states` | `[B, T, H]` = `[B, T, 2560]` | bf16 | target 末层 hidden；**当前 recipe 未消费**（保留字段；recompute 模式不产出。原计划 S3 k=1 golden 对拍用，🔷 未接入） |

---

## 阶段 S2 — Actor 构建 + Rollout ✅（actor + 采样 + top-K 候选，smoke+E2E 通过）

> IP-2（actor model 构建）已并入 S2：verl 里 rollout = 用 actor 权重采样，二者一体（见 `DSpark-OPD.md` S2）。
> 实测 E2E（`num_anchors=32`, `K=16`）：`Qwen3DSparkModel` FSDP actor（bf16、`fsdp_size=1`、`use_orig_params=True`）→ 块并行采样 + top-k → `rollout_tokens (B,32,7)`、`rollout_logp_draft (B,32,7)` finite ≤0、`rollout_student_top_k_ids/logp (B,32,7,16)`（top-1 == 采样 token、logp 降序）。rollout 次数 `n` 在 batch 维（verl `rollout.n`），rc=0。

### 输入（S2 真正消费的子集）

| 张量 | shape | dtype | 用途 |
|---|---|---|---|
| `input_ids` | `[B, T]` | long | 取 anchor token（每块首位） |
| `loss_mask` | `[B, T]` | long | 采 anchor（仅 `>0` 处）；派生 `eval_mask` |
| `target_hidden_states` | `[B, T, 12800]` | bf16 | 块并行 backbone 的 context K/V（经 `fc`+`hidden_norm`） |

### 内部中间张量（块并行 forward，复用训练路径 `Qwen3DSparkModel.forward`）

| 张量 | shape | dtype/类型 | 语义 |
|---|---|---|---|
| `anchor_positions` | `[B, A]` | long | 每样本采的 anchor 位置（仅 `loss_mask>0`） |
| `block_keep_mask` | `[B, A]` | bool | 有效 anchor 掩码（短样本不足 A → 部分 False） |
| `noise_embedding` | `[B, A*blk, H]` | bf16 | 每块首位=anchor token emb，其余=`mask_token_id` emb |
| `draft_position_ids` | `[B, A*blk]` | long | 每块 = `anchor+1 .. anchor+blk` |
| `dspark_attn_mask` | **`BlockMask`**（逻辑 `[B, A*blk, T+A*blk]`） | FlexAttention 对象 | 块对角可达性：每块只 attend context `[0,anchor)` + 同块 draft 位；无效块整行关闭。KV 轴 = `[context(T 列) | draft(A*blk 列)]`。**非数据张量**，由 `mask_mod` 函数定义、按块稀疏跳过 |
| `output_hidden` (4D) | `[B, A, blk, H]` | bf16 | 块并行主干输出 |
| `base_logits` | `[B, A, blk, V]` | float | lm_head 输出（pre-markov）；**仅用于 eval 一致性对拍**（采样与 top-k 都用 markov 校正后分布，非此） |
| `corrected_logits` | `[B, A, blk, V]` | float | markov 校正后分布（`base_logits` + markov 偏置）；**采样、`logp_draft`、top-k 候选均基于它**（三者同源） |

> **关于 `B`**：下文所有 `B` 均指 **rollout 后 batch**（`= B_prompt × rollout.n`）。rollout 次数 `n` 已吸收进 `B`（`gen_batch.repeat(rollout.n)`），**不作为独立张量维**；同一缓存样本的 n 个副本各自独立采 anchor。

### 输出（S2 产物 → S3/S4）

| 张量 | shape | dtype | 语义 |
|---|---|---|---|
| `tokens` (ỹ) | `[B, A, blk]` | long | ★ 草稿采样 token（markov 块内自回归） |
| `logp_draft` | `[B, A, blk]` | float | 草稿对采样 token 的 logπ；`≤0`、有限（无效块已排除） |
| `student_top_k_ids` | `[B, A, blk, K]` | long | ★ top-K 候选 id，`topk(corrected_logits, K)` 顺手取（no_grad，只取 id）→ 供 S3 teacher 打分。✅ 已实现 |
| `student_top_k_logp` | `[B, A, blk, K]` | float | no_grad 候选 logp（降序；诊断/备用；带梯度版在 S4 重算）。✅ 已实现 |
| `anchor_positions` | `[B, A]` | long | 透传，供 teacher 对齐 |
| `block_keep_mask` | `[B, A]` | bool | 有效块掩码 |
| `eval_mask` | `[B, A, blk]` | bool | 块内哪些位置计入信号（`loss_mask` 派生；充当 `response_mask`） |

> **无效块（`block_keep_mask=False`）不参与采样**（避免块对角 mask 下 softmax 全 -inf 的 NaN）；其采样输出为占位，被 `eval_mask` 排除。
> **top-k 候选 id：DSpark 在 S2 顺手取，无需独立前向**。Rethink-OPD 靠 rollout 后一次独立学生前向（`compute_log_prob(top_k)`）出候选，因 vLLM 不留 logits；而我们 `block_rollout` 采样时已算出 `corrected_logits`，**`student_top_k_ids` 在 S2 顺手 `topk(corrected_logits, K)` 取**（no_grad，只取 id，零额外前向，见「前向次数账」）。候选**取自 markov 校正后分布**（与采样同源，无歧义）。
> **带梯度的候选 logp 在 S4 重算**：S2 的 `student_top_k_logp` 是 no_grad 的（诊断/备用）；loss 需要的**带梯度**学生 logp 在 S4 训练前向时于这些固定候选 id 上重算（PPO 惯例，见 §S4）。
> rollout 次数 `n` 由 verl `gen_batch.repeat(rollout.n)` 在 batch 维实现，block_rollout 本身 R-agnostic。

**S2 必做检查**（smoke，`scripts/opd/s2_smoke.py`）✅ 已通过（实测 num_anchors=64，2 样本 real_len=[226,244]）：
- **eval 一致性**：批量 flex 块 logits vs eval 单块 sdpa 内核（`forward_dspark_draft_block`），同样本同 anchor。实测 5 anchor **相对最大偏移 relmax=6.1e-3~7.4e-3、贪心 token 全一致**（绝对 `max|Δ|`=0.14~0.19，logit 尺度~25；纯 flex-vs-sdpa bf16 噪声）。
- **批不变性**：样本单独跑 vs padded batch 跑。实测**相对均值误差 1.6e-3、argmax 100%**（bf16/flex tiling 噪声，非 padding 泄漏）。
- **无 NaN**：有效块 `logp_draft` 有限且 ≤0 ✅。
- **anchor 合法性**：有效块 anchor ∈ `loss_mask>0` ✅。
- **确定性**：同 seed 两次一致；temp=0 两次一致 ✅。
- 🔷 **FSDP 透明性（可选补充，未做）**：同样本同 anchor 下，**裸模型**（smoke 路径）与 **FSDP-wrapped actor**（verl worker attach 给 rollout）的 block logits `allclose`。当前 FSDP actor 用与 smoke 相同的 bf16+flex 配置且 `fsdp_size=1` 无分片，数值一致性有强保证；严格证明可另补对拍脚本。证明 FSDP 包裹的非分片副作用（mixed_precision/use_orig_params）不改变数值。**FSDP 配置见 `DSpark-OPD.md` §2.6.5**：`fsdp_size=1`（每卡完整参数，避免 FULL_SHARD 子方法分片）+ `use_orig_params=True`（因冻结 embed/lm_head，必需）+ `MixedPrecision(bf16)`，与 DeepSpec 训练（`base_trainer.py`）对齐。

> 度量定义：`max|Δ|`=块 logits（`blk×V`）逐元素 `|ours−eval|` 最大值；`relmax`=`max|Δ|/|logit|max`。

---

## 阶段 S3 — Teacher 打分（路线 A，块对角 online 前向）✅（`score_blocks_flat`，展平等价性通过）

> 实测（`teacher_scoring.py`，num_anchors=8，2 样本 real_len=[226,244]）：`score_blocks_flat`（一次 target forward + 4D 块对角因果 mask + 真实 position_id）vs `score_blocks_reference`（逐 block 独立前向）——**fp32 all-K `max|Δ|=1.45e-4`（结构等价）**；bf16 argmax 100%、top-1/student-weighted `mean|Δ|≈0.02`（长序列 bf16 噪声）。修了一个真实 bug：ctx key 原 `<=anchor` 使 anchor token 双计（既 ctx key、又 block-key j=0），bf16 top-1 `max|Δ|` 曾达 7.0 → 改 `<anchor` 后降到 1e-4。`compute_rm_score` 在 `DSparkRewardModelWorker` 整体覆写（非 verl `_forward_micro_batch`），gate=`teacher`。

> **⚠️【批处理化，效率修复】`score_blocks_flat` 从"逐样本循环"改为"batch 维一次 forward"**：
> - **背景**：初版 `for b in range(B):` 逐样本各跑一次 `target([1,T+Q], [1,1,T+Q,T+Q] mask)`。teacher 是 FULL_SHARD，**每次 forward 都 all-gather 一次 4B 参数** → B 个样本 = all-gather B 次 + B 次 kernel 启动，是 teacher 阶段 GPU 空置/慢的主因。
> - **改法**：一次 `target([B,T+Q] ids, [B,1,T+Q,T+Q] mask, [B,T+Q] pos)` 算完整个 batch。`T = batch 内 max real_len`（`dspark_collate_fn`，实测 ~244，**非** `max_prompt_length=4096`），故 `T+Q≈468`，4D mask `[B,1,468,468] bf16` 在 B=16 时才 ~7MB，显存无压力。teacher all-gather 降到 **1 次**。
> - **为何仍与逐条一致（等价性不变）**：① 每样本的 4D 块对角因果 mask 逻辑逐样本独立构造后堆进 batch 维（`allow_ctx[b]=key_pos<anchor_i[b]`、块对角、块内因果），batch 维之间在 attention 里天然隔离（不同 batch 行不互相 attend）——这是标准 batched attention 语义，不引入跨样本串扰；② **padding 天然被排除**：anchor 恒在真实 token 内（`sample_anchor_positions` 只采 `loss_mask>0`），故 `allow_ctx=key_pos<anchor_i` 使 padding 列 `[real_len_b,T)`（`key_pos≥real_len_b>anchor_i`）恒 False，block query 从不 attend padding，**无需额外 attention_mask**；③ RoPE 按真实 `position_id`（`anchor_i+j`），batch 化不改变。
> - **必做校验（复用 S3 golden 精神）** ✅ **已验证**：batched `score_blocks_flat` vs `score_blocks_reference` 仍逐 block 一致（test A 数值与逐样本版完全相同）；且 **B=2 一次打分 vs 每样本 B=1 单独打分 `max|Δ|=0`（bit-exact，变长 real_len=226/244 故 sample0 带 padding）**——证明 batch 维零串扰、padding 零泄漏（s3_smoke test E）。S4 smoke 数值不变（下游 reward/loss 未受影响）。

> **⚠️ 维度语义（关键，S3 实现前必读）**：`A*R*blk` **可以物理展平成一条张量序列**（draft 侧 `create_dspark_attention_mask` 已如此），前提是同时满足两个条件；否则展平会出错：
> - **条件 (a) 块对角 mask 隔离 `A`**：`A`（32 anchor）是同一 response 上不同位置的**平行反事实分支**，**必须完全隔离**（绝不互相 attend）。用 `dspark_mask_mod`（`q_block_id==kv_block_id`）精准控制每 token 的 attend 对象即可隔离。**若展平后用标准因果 attention（不隔离）→ 跨 anchor 串扰、打分错误。**（rollout 次数 n 在 batch 维、天然隔离，无关此处。）
> - **条件 (b) position_id 用"真实序列位置"而非展平物理下标**：DSpark 用 **RoPE**（`Qwen3RotaryEmbedding`），编码按 `position_id` 而非张量物理下标。`create_position_ids` 给每个 block 赋 `anchor+1..anchor+blk`（**该 block 在原序列的绝对位置**）。故展平后各 block 仍拿到各自真实位置的 RoPE 相位，**物理相邻不会错乱**；配合 (a) 的隔离，相对位置只在可 attend 的 token 间生效。
> - 满足 (a)+(b)，展平在数值上与"逐 block 独立算"**等价**。这就是 §2.4.1「批量化 `verify_draft_tokens`」的正确实现：一次 forward 算完全部 A×blk block。
> - **blk 维**：块内**因果**（位置 k attend `ỹ_1..ỹ_{k-1}` + context `[0,anchor)`）。
> - **⚠️ teacher 块内因果 vs draft 块内双向**：draft `_forward_backbone` 用 mask token 并行预测整块（`is_causal=False`，块内全连接）；teacher 用真实采样 token 做**因果**打分（`verify_draft_tokens` 走 causal）。故 **S3 的块对角 mask 块内必须是下三角（因果），不能照抄 draft 的双向块 mask**。

### top-k 候选来源（S2 已产出，S3 直接消费）

`student_top_k_ids/logp` **在 S2 rollout 时顺手取**（`topk(corrected_logits, K)`），**不需 S3 再前向学生**——这是 DSpark 相对 Rethink-OPD 省下的那次前向（Rethink-OPD 因 vLLM 不留 logits 才需独立 `compute_log_prob(top_k)`，见「前向次数账」）。S3 直接拿 S2 的 `student_top_k_ids` 作 teacher 打分输入。

> **【已锁定，无歧义】top-k 候选、token 采样、`logp_draft` 三者同源——都基于 markov 校正后分布 `corrected_logits`**（学生实际输出分布，与 Rethink-OPD `only_stu` 取学生真实分布 top-k 一致）。**不用 pre-markov `base_logits`**（否则候选集与采样分布错位，加权 KL 出错）。代码事实：`sample_block_tokens` 采样即用 `apply_step_logits` 校正后（`markov_head.py:78`）。

### Teacher 打分输入

| 张量 | shape | dtype | 用途 |
|---|---|---|---|
| `input_ids` | `[B, T]` | long | target 对真实 context 做因果 prefill |
| `tokens` (ỹ) | `[B, A, blk]` | long | 追加为 query，条件于各块采样前缀打分 |
| `student_top_k_ids` | `[B, A, blk, K]` | long | teacher 需在这些 top-K 候选上给 logπ（top-k 稠密） |
| `anchor_positions` | `[B, A]` | long | 定位每块在序列中的位置 |
| `block_keep_mask` | `[B, A]` | bool | 排除无效块 |

### 输出

| 张量 | shape | dtype | 语义 |
|---|---|---|---|
| `logp_target_on_topk` | `[B, A, blk, K]` | float | teacher 在学生 top-K 候选上的 logπ（`only_stu`：`T_on_S`）；OPD 加权 KL 的 teacher 侧 |
| （可选）`logp_target_sampled` | `[B, A, blk]` | float | teacher 在实际采样 token 上的 logπ（单样本对照/诊断用）|

> k=1（块首位）的 teacher 分布应与缓存 `target_last_hidden_states[anchor]` 过 lm_head 一致（S3 黄金对拍）。

**S3 必做检查（smoke，`scripts/opd/s3_smoke.py`）**：
- **★ 展平等价性（本阶段最强校验，先验证再用）** ✅：**展平批量打分**（A×blk 展平 + 块对角因果 mask + 真实 position_id，一次 target forward = `score_blocks_flat`）vs **逐 block 独立打分**（对每个 block 单独跑一次 target，causal = `score_blocks_reference`），二者 `logp_target` 一致。**黄金证明**——fp32 all-K `max|Δ|=1.45e-4`（结构等价），bf16 argmax 100%。**⚠️ 关键陷阱（已修）**：展平时 ctx key 必须 `key_pos < anchor`（**严格小于**）——anchor token 由 block-query 侧 `j=0` 提供，若 ctx key 用 `<=anchor` 会把 anchor 双计入 attention，扰动 softmax（bf16 `max|Δ|` 曾达 7.0）。
- **块对角隔离** ✅：改某 block 的采样 token，其它 block 的 `logp_target` 不变（anchor 间无串扰）。
- **块内因果** ✅：改块内最后一个 token 不影响更早位置的 `logp_target`（验证块内是因果而非双向）。
- **无 NaN** ✅：有效块 `logp_target` 有限且 ≤0。
- **★ batch 等价性（批处理化后新增，本次必验）**：**batched `score_blocks_flat`（B 样本一次 forward）** vs **`score_blocks_reference`（逐样本逐 block）**，逐样本逐 block `logp_target` `allclose`（fp32 结构等价，bf16 argmax 100% + top-1/student-weighted `mean|Δ|` 同量级）。★额外**跨 batch-size 一致性**：同一批样本,`B=n` 一次打分 vs 每样本单独 `B=1` 打分,结果 `allclose`（证明 batch 维无串扰、padding 不泄漏）。用**变长样本**（real_len 不等，故意触发 padding）跑此项。
- 🔷 **k=1 黄金对拍（暂缓至 S4）**：块首位 teacher 分布 == 缓存 `target_last_hidden_states[anchor]` 过 lm_head（路线 A 的 online prefill 与缓存自洽）；当前正确性已由展平等价性 + 隔离 + 块内因果三项共同锁定，缓存自洽对拍待 S4 接入缓存 last_hidden 时补。

> 度量同 S2：报告 `max|Δ|` / `relmax`（相对 logit 尺度）/ argmax 一致率。

---

## 阶段 S4 — Reward → Advantage → Loss（`update_dspark_opd`；训练前向经 `FSDP.forward`）

> 落地为**单个 actor 方法 `update_dspark_opd`**（非 verl 三段式）。**★ 训练前向经 `Qwen3DSparkModel.forward`（OPD 分支：传入固定 `anchor_positions`/`block_prev_tokens`）→ `FSDP.forward`**，使 FSDP1 的 pre-forward hook 注册、多卡梯度跨 rank all-reduce（**不得用子方法直接调用绕过**——初版 `dspark_block_train_forward` 那样做导致 root-unit 训练头梯度多卡不同步，CRITICAL，已修；见 `DSpark-OPD.md` §S4 note + memory `dspark-opd-multigpu-grad-sync-bug`）。`draft_logits`(=corrected) 上取固定候选带梯度 logπ；`token_reward_direct` `advantage==rm*mask`；3D dual-clip PG；confidence 目标用 **top-K 支撑近似** `Σ_k min(p_d,p_t)`（复用 S2/S3 no-grad top-k，零新增搬运）。单卡实测（`s4_smoke.py`）：展平可逆、前向复现 rollout corrected（`mean|Δ|≈0`）、backward 后可训练 grad 有限>0、冻结无梯度。多卡梯度一致性由 `s4_grad_sync_smoke.py`（torchrun 2-rank，all-gather 比对训练头 grad `allclose`）覆盖。

### 块结构 → 序列轴展平（§2.6.4 caveat #2）

verl 的 `token_reward_direct` + 3D top-k `compute_policy_loss_vanilla` 期望 `[B, response_len, K]`；
DSpark 是 `[B, A, blk]`（+ top-k 维 K）。展平：把 `A*blk` 摊平进 `response_len`，
保留 top-k 维 K；`eval_mask` 充当 `response_mask`。（rollout 次数 `n` 已在 `B` 维，天然并入 batch。）

| 张量 | shape | dtype | 语义 |
|---|---|---|---|
| `rm`（token reward） | `[B, A, blk, K]` → 展平 `[B, A*blk, K]` | float | top-k 稠密：`w_j·(logp_target_on_topk − student_top_k_logp)`；`w_j`=`softmax_K(student_logp)`（`student_p`）|
| `advantage` | 同 `rm` | float | `token_reward_direct`：`= rm`（mask 广播） |
| `response_mask` | `[B, A*blk]` | bool/长 | 由 `eval_mask` 展平 |
| `logp_draft`（带梯度重算） | `[B, A*blk, K]` | float | policy loss 用；在 top-K 候选上带梯度重算学生 logπ（`logp_on_topk_ids(corrected_logits, top_k_ids)`）|
| `accept_rate`（confidence 目标） | `[B, A, blk]` | float | **top-K 近似** `Σ_k min(p_draft_k, p_target_k)∈[0,1]`（detach）；BCE vs 带梯度 `confidence_pred` |
| `pg_loss` | 标量 | float | dual-clip PG（对 K 求和，`token-mean`）；总 loss = `pg_loss + confidence_head_alpha·conf_bce` |

> 展平必须**可逆**（`flatten↔unflatten` 逐元素相等，实测 `torch.equal`），否则 reward 会错位到别的块位置（S4 关键 smoke，已过）。
> `n` 个 rollout（在 `B` 维）逐条独立，loss 直接平均，不做 GRPO 组内归一（`token_reward_direct` 语义，见 §1.7b）。
> block 内 `loss_decay_gamma=4.0` 衰减权重（`block_decay_weight_mask` = `eval_mask × exp(-pos/γ)`）**同时作用于 PG 与 confidence BCE**：作为 `response_mask` 传入 `token_reward_direct`（`advantage = rm × decay_mask`）+ `compute_policy_loss_vanilla`（`token-mean` 变成 decay 加权均值），复刻 SFT `loss_weight_mask` 语义（§2.5.4）。
> **micro-batch 切分 + 梯度累积** ✅：`update_dspark_opd` 按 `ppo_micro_batch_size_per_gpu` `data.split()` 切 micro，逐 micro forward+`(loss×1/n_micro).backward()` 累积，末尾一次 `_optimizer_step`（照搬 verl **静态**分支 `dp_actor.py:797-819`）。`n_micro = B_recv // ppo_micro_batch_size_per_gpu` = verl `gradient_accumulation`。聚合含 token-mean-vs-`1/M` 近似（各 micro token 数不等时有偏差）。**smoke 覆盖**：`s4_smoke.py` test F 验证「分母相等时 micro 累积梯度 == 整批梯度」（`max|Δgrad|=6.8e-3`）。**注**：Rethink-OPD OPD 实际用 `use_dynamic_bsz`（`loss_scale=样本数/mini`），我们对齐的是 verl 静态分支——因块结构 per-sample 等长、`micro=1` 时两式严格相等，取舍见 `DSpark-OPD.md` §S4 note。

---

## 维护规则

1. **每阶段实现/修改时**，先对照本表核对该阶段的输入/输出 shape、dtype、语义；smoke-test 的断言应引用本表的数值。
2. **张量设计变化时**（如 rollout 是否在 S2 就展平、是否引入 top-k 维 K），先改本文件，再改代码，保持契约与实现一致。
3. **状态标记**随实现推进更新（🔷→✅），并在对应行附「实测值」示例（如 S1：T=226、`input_ids[:8]=[151644,...]`）。
4. 主文档 `DSpark-OPD.md` §2.9 数据流图与本文件必须一致；若冲突，以本文件（更细粒度）为准并回改 §2.9。
