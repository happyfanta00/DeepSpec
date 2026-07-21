# T5 (sglang rollout) train_step —— 逻辑 Tensor 契约（合并方案）

> 按**逻辑张量**梳理（不按代码打包容器）。每步列 Input / Output 的每个独立张量：
> 形状、含义、来源→去向。合并方案：Step 4 用**一次朴素因果 teacher 前向**同时产出
> draft 用的 hidden 与 loss 用的 teacher top-k。

## 符号

| 符号 | 含义 | 值 |
|---|---|---|
| `B` | 每 rank 样本数 = `train_batch_size/dp × rollout_n` | 32 |
| `pᵢ` / `rᵢ` | 第 i 条样本的 prompt 长度 / response 长度（**变长**，样本间不等） | 数据依赖 |
| `T` | 序列全长 = `maxᵢ(pᵢ+rᵢ)`（prompt++response 右 padding 后） | 数据依赖 |
| `R` | response 段最大长度 = `maxᵢ(rᵢ)`（≤T） | 数据依赖 |
| `A` | 每样本 block 数（`dspark_max_anchors_cap` 封顶） | ≤256 |
| `blk` | block_size | 7 |
| `V` | 词表 | 151936 |
| `L` | teacher 中间层数（`target_layer_ids`） | 5 |
| `H` | teacher hidden_size | 2560 |
| `K` | draft 候选 top-k（`log_prob_top_k`） | 16 |
| `Kt` | teacher top-k（`dspark_teacher_top_k`） | 64 |
| `micro` | `ppo_micro_batch_size_per_gpu`（Step4/5 batch 维分块粒度） | 8 |

## Block 内部结构（双流）

一个 block（一轮 decode）在 blk 个 slot 上的语义：

```
slot:   0 .. n_acc-1        n_acc              n_acc+1 ..
        └── accept 段 ──┘   └reject(boundary)┘  └── discarded ──┘
        draft 提议被接受     draft 提议被拒、       block_size 内
        的 token            target 纠正的 token    未用到的尾部
```

- `n_acc = block_lengths - 1`（该轮 accept 的 token 数）。
- **accept 流**：slot `0..n_acc-1`，draft 提议被接受的位。
- **reject 流**：slot `n_acc`（**恰一个** boundary 位），draft 提议被拒、target 提交纠正 token 的位。
  - 仅当该 block **有 boundary** 且 `n_acc < blk` 时存在；stop-trim 的末轮 partial block 无 boundary → 无 reject 位。
- **两流的 token 都在 response 里**（accept token + boundary 纠正 token 均已提交），故两流的 draft/teacher 分布都可由 teacher-force 前向重算，**无需 hook、无需被拒 token id**（Draft-OPD 双流：两流分别进 loss，可各自加权/不同处理）。
- discarded 段（slot `>n_acc`）：draft 被拒的其它提议未导出，**不进 loss**。

---

## Step 1 — sglang rollout（现场投机解码）

> 变长张量：每条样本独立、长度不等，记为 `B × [·]`（B 条，每条第 2 维长度为 `pᵢ`/`rᵢ`）。padding 到规整 `[B,·]` 发生在 Step 2。

**Input**

| 张量 | 形状 | 含义 | 来源 |
|---|---|---|---|
| `prompt_ids` | `B × [pᵢ]` | golden prompt token（去 padding，第 i 条长 `pᵢ`） | DSparkPromptDataset（做法X）|

**Output**（逻辑上是两条独立流，代码里混在同一个 `gen_outs` dict）

| 张量 | 形状 | 含义 | 作用（去向） |
|---|---|---|---|
| `output_ids` | `B × [rᵢ]` | 第 i 条 response 的生成 token 序列（长 `rᵢ`） | → Step 2 拼成序列 |
| `accept_state` | `B × [rᵢ]` | 第 i 条 response 的**投机解码接受轨迹**（逐 output-token：0=accept,1=commit boundary,2=prefill seed），长度 == `rᵢ` | → Step 3 **重建 block plan**（哪些位是 anchor、每 block 多长）；本身**不进任何前向**，纯粹是"draft 真实走过的 block 结构"的记录 |

> `output_ids` 是"生成了什么 token"，`accept_state` 是"这些 token 是按怎样的 block 结构投机出来的"——两者正交，作用不同，逻辑上分开。

---

## Step 2 — 拼序列 + 右 padding（`rebuild_padded_batch`，无前向）

**Input**：`prompt_ids`（Step1）、`output_ids`（Step1）、`accept_state`（Step1）

**Output**

| 张量 | 形状 | 含义 | 作用（去向） |
|---|---|---|---|
| `input_ids` | `[B,T]` | prompt++response，右 padding | → Step4 teacher 前向、Step5 draft 前向的序列输入 |
| `attention_mask` | `[B,T]` | 真实 token=1，padding=0 | → Step4/5 屏蔽 padding |
| `loss_mask` | `[B,T]` | response 段=1，prompt+padding=0 | → 标记监督区间（draft 前向）|
| `accept_state`(padded) | `[B,R]` | Step1 accept_state 右 padding（pad=-1，response 索引对齐） | → Step3 重建 block plan |
| `response_lengths` | `[B]` | 每条 response 真实长度 | → Step3 切有效 accept_state |
| `prompt_lengths` | `[B]` | prompt 长度（=response 在 input_ids 的起点） | → Step3 把 response 索引换算成 input_ids 全局索引 |

---

## Step 3 — 重建 block plan（`reconstruct_block_plan`，纯 index，无前向）

**Input**：`input_ids`（Step2）、`accept_state`(padded)（Step2）、`response_lengths`（Step2）、`prompt_lengths`（Step2）、`block_size`

**Output**

| 张量 | 形状 | 含义 | 作用（去向） |
|---|---|---|---|
| `anchor_positions` | `[B,A]` | 每 block 的 anchor 在 input_ids 的全局 index | → Step4 定位 block 打分位；Step5 draft 前向 |
| `block_keep_mask` | `[B,A]` bool | 有效 block（短序列尾部=False） | → Step4/5 屏蔽无效 block |
| `block_tokens` | `[B,A,blk]` | 每 block 内的真实 token（accept 段 + reject/boundary 位，真实轨迹，非重采） | → Step5 构造 `block_prev_tokens` teacher-force |
| `slot_type` | `[B,A,blk]` int8 | 每 slot 的**双流类型**：`0`=accept，`1`=reject(boundary)，`-1`=discarded/padding（不进 loss） | → Step5 双流 loss 掩码（区分 accept/reject 位）|

**双流掩码由 `slot_type` 派生**（Step5 用）：
- `accept_mask = (slot_type == 0)` — accept 流。
- `reject_mask = (slot_type == 1)` — reject 流（每 block ≤1 个）。
- `eval_mask = (slot_type >= 0)` — 进 loss 的全部位（accept ∪ reject）。

> 取代原"选项 B 单一 accept `eval_mask`"。reject 位的 `block_tokens` = boundary 纠正 token（在 response 内），故其 draft/teacher 分布在 Step4/5 的 teacher-force 前向里天然算得（该位以 anchor + 前 `n_acc` 个 accept token 为前缀），无需被拒 token id。

---

## Step 4 — 合并 teacher 前向（`teacher_causal_hidden_and_topk`，`@no_grad`，按 micro 切分）

一次朴素因果前向：只跑 backbone（不调 `ForCausalLM.forward`，不算全序列 lm_head）+ hook 抓中间层；**只在 block 位**过 lm_head 取 top-k。

**Input**

| 张量 | 形状 | 含义 | 来源 |
|---|---|---|---|
| `input_ids` | `[B,T]` | 序列 | Step2 |
| `attention_mask` | `[B,T]` | 屏蔽 padding | Step2 |
| `anchor_positions` | `[B,A]` | block anchor 全局 index | Step3 |
| `block_keep_mask` | `[B,A]` | 有效 block | Step3 |
| `target_layer_ids` | `[L]` | 抓哪些中间层 hidden | 模型常量 |
| `teacher_top_k` | int=Kt | teacher 候选数 | config |
| `block_size` | int=blk | | config |

**内部量**（非对外输出）

| 量 | 形状 | 说明 |
|---|---|---|
| block 位全局 index `p[b,r,j]=anchor[b,r]+j` | `[B,A,blk]` | clamp 到 T-1 |
| block 位 last-hidden（gather） | `[micro,A,blk,H]` | 只取 block 位 |
| block 位 lm_head logits | `[micro,A,blk,V]` | **只此处、只 block 位**（非全 T×V）|

**Output**

| 张量 | 形状 | 含义 | 作用（去向） |
|---|---|---|---|
| `target_hidden_states` | `[B,T,L*H]` bf16 | 中间层 hidden，draft 的 cross-attn KV 上下文 | → Step5 draft 前向 |
| `t_ids` | `[B,A,blk,Kt]` | teacher 自己 top-64 的 token id | → Step5 对齐 draft 候选 |
| `t_logp` | `[B,A,blk,Kt]` | 上述 id 的真 full-vocab logπ（logsumexp） | → Step5 teacher 侧 loss 信号 |

---

## Step 5 — draft 前向（Phase-3，grad，按 micro 切分）

draft block 前向，产出带梯度的候选分布；不含 loss。

**Input**

| 张量 | 形状 | 含义 | 来源 |
|---|---|---|---|
| `input_ids[sl]` | `[micro,T]` | 序列 | Step2 |
| `target_hidden_states[sl]` | `[micro,T,L*H]` | cross-attn 上下文 | Step4 |
| `loss_mask[sl]` | `[micro,T]` | 监督区间 | Step2 |
| `anchor_positions[sl]` | `[micro,A]` | block anchor | Step3 |
| `block_keep_mask[sl]` | `[micro,A]` | 有效 block | Step3 |
| `block_prev_tokens` = `[anchor_tok, block_tokens[:, :, :blk-1]]` | `[micro,A,blk]` | teacher-force markov 输入（含 accept + reject 位真实 token） | Step3（`block_tokens`）|
| `t_ids[sl]` / `t_logp[sl]` | `[micro,A,blk,Kt]` | teacher top-64 | Step4 |

**Output**

| 张量 | 形状 | 含义 | 作用（去向） |
|---|---|---|---|
| `draft_logits` | `[micro,A,blk,V]` grad | draft block 预测 logits（accept + reject 位都算） | → 取 draft top-K / teacher top-K 上的 draft logπ |
| `confidence_pred` | `[micro,A,blk]` grad | 接受率预测头 | → Step6 confidence BCE |
| `d_ids` / `S_grad`（`draft_topk_logp`） | `[micro,A,blk,K]` | draft top-K id（no-grad）+ 该 id 真 logπ（grad） | → Step6 **reject 流** reverse-KL 支撑集 |
| `T_on_S`（`align_teacher_to_draft(d_ids, t_ids, t_logp)`） | `[micro,A,blk,K]` | teacher 在 draft K 个 id 上的 logπ（命中 t_top-Kt 用真值，否则 min），no-grad | → Step6 **reject 流** teacher 侧对齐信号 |
| `S_on_T`（`logp_on_topk_ids(draft_logits, t_ids[...,:K])`，logsumexp） | `[micro,A,blk,K]` grad | **draft 在 teacher top-K id 上的 full-vocab logπ**（新增，forward KL 用） | → Step6 **accept 流** forward KL 的 draft 侧 |

> K = `log_prob_top_k`（当前 16），forward/reverse 同一个 K，只是候选来源不同。forward KL 的 teacher 侧直接复用 Step4 的 `t_ids[...,:K]` / `t_logp[...,:K]`（teacher top-`Kt` 的前 K，topk 已降序）——无需新前向；新增的只有 `S_on_T`（draft 前向后在 teacher top-K id 上多一次 gather）。

---

## Step 6 — loss 计算（双流，按 micro 切分）

消费 Step5 候选 + Step3 双流掩码；分别聚合 accept / reject 流。

**Input**

| 张量 | 形状 | 含义 | 来源 | 用于 |
|---|---|---|---|---|
| `S_on_T` | `[micro,A,blk,K]` grad | draft 在 teacher top-K id 上 logπ | Step5 | accept 流 forward KL |
| `t_logp[...,:K]` | `[micro,A,blk,K]` no-grad | teacher top-K logπ | Step4 | accept 流 forward KL |
| `S_grad` | `[micro,A,blk,K]` grad | draft top-K logπ | Step5 | reject 流 reverse KL |
| `T_on_S` | `[micro,A,blk,K]` no-grad | teacher 在 draft top-K id 上 logπ | Step5 | reject 流 reverse KL |
| `confidence_pred` | `[micro,A,blk]` grad | 接受率预测头 | Step5 | confidence BCE |
| `slot_type[sl]` | `[micro,A,blk]` int8 | 双流类型（0=accept,1=reject,-1=不进 loss） | Step3 | 派生掩码 |

**派生掩码**（从 `slot_type`）：`accept_mask=(slot_type==0)`、`reject_mask=(slot_type==1)`、`eval_mask=(slot_type>=0)`。

**双流 loss 定义**（2026-07-20 起 accept 流 = forward KL；K = `log_prob_top_k`，forward/reverse 同一个 K）：
- **accept 流** `forward_kl`：`p_t = softmax_K(t_logpK)`（no-grad）；`L = Σ_j p_t(j)·(t_logpK_j − S_on_T_j)`（teacher top-K 支撑，权重 teacher 侧）。
- **reject 流** `reverse_kl`（不变）：`p_θ = softmax_K(S_grad)`；`L = Σ_j p_θ(j)·(S_grad_j − T_on_S_j)`（draft top-K 支撑，权重 draft 侧）。

**Output**

| 张量 | 形状 | 含义 | 作用 |
|---|---|---|---|
| `loss` | scalar | `w_fwd·forward_kl(accept_mask) + w_rej·reverse_kl(reject_mask) + w_conf·confidence_bce(eval_mask)` | → backward |
| `loss_metrics` | dict[scalar] | 分项：`forward_kl_loss`(accept)、`reject_kl_loss`(reject)、`confidence_loss` 等 | → 指标聚合 |

> accept 流用 forward KL（mass-covering，权重=teacher）、reject 流用 reverse KL（mode-seeking，权重=draft），按 `accept_mask`/`reject_mask` 分别聚合、各自加权。权重 `w_fwd/w_rej/w_conf` 由 config（`dspark_loss_forward_kl_weight` / `dspark_loss_reject_kl_weight` / `dspark_loss_confidence_weight`）定。原 `reverse_kl`(accept 流) 默认关（`dspark_loss_reverse_kl_weight=0`）；若 >0 则 accept 流 forward+reverse 叠加。

---

## Step 7 — T6 权重回灌 + 指标

**Input**：训练后的 draft state_dict（rank0）
**Output**：`draft_weights_pushed`(int) + 标量 `loss/reverse_kl_loss/reject_kl_loss/confidence_loss/grad_norm/n_micro`。
