# DSpark Draft on SGLang：投机解码集成设计与操作指南

> **一句话目标**：让 SGLang 用 **DSpark** draft model 做投机解码 rollout 并接进 verl，使
> DSpark-OPD 训练的 response **由 SGLang 投机解码现场产生**（替换现有 cache response），同时把
> rollout 过程中的 draft 信息导出供后续实验。
>
> **本文档是这项工作的唯一权威文档。** 早期的 HANDOVER、rejection 正确性调查、anchor 捕获设计
> 三份文档的有效结论已全部并入本文，原文档已删除。
>
> 分支：`feat/dspark-sglang-rollout`（main 保持干净，全部验证通过后由用户手动合并）。
>
> **本文档组织逻辑**：
> 1. **相关项目背景理解**（§1）——DSpark/DFlash draft 架构、markov head、upstream 原生 DSPARK 路径。
> 2. **我们的需求与可行性分析**（§2）——正确性标准、fork 为何不达标、upstream 为何可行、评测金标准。
> 3. **代码开发方案**（§3 起，Stage-M0/M1/M2/M3）——每阶段的**具体做法 / smoke-test 方法与结果 /
>    用户手动测试方法**三段式。
> 4. **资源与操作速记 + 历史路线**（§8、附录）。

---

## 0. 一页速览（TL;DR）—— 2026-07-16 最新状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| **Stage-M0** | upstream sglang env 安装（§3）| ✅ 完成 |
| **Stage-M1** | 正确性 + 接受率 + 加速比三项验证（§4）| ✅ 全部达成 |
| **Stage-M2** | verl rollout 接 upstream DSPARK sglang（HYBRID 共卡 + 融合 train_step 内 sleep/wake）+ draft 信息最小契约导出 + draft 权重回灌；拆 6 子任务 + 集成（§5）| ⏳ 进行中（T1+T2 ✅，均含 server 待测）|
| **Stage-M3** | DSpark-OPD 训练闭环 + 信息导出固化（§6）| ⏳ 待做 |

**最重要的方向决策**：改基座到 upstream sglang，使用其**原生 DSPARK 投机解码路径**，放弃在 fork
`sglang-dflash` 上手写 markov 串行采样（原因见 §2 可行性分析）。

**Stage-M1 三项结论**：
1. **正确性（内核级等价 + 无损）**：upstream `reject_sampling.chain_speculative_sampling_triton`
   与本 repo rejection sampling 输出分布一致（Δ≤0.007），候选从 q 重采样后边缘分布收敛到 target
   p0（Δ≤0.004）→ **等价 + 无损**。
2. **接受率对拍（≤2% 吻合）**：gsm8k、seed 980406、temp=1.0、enable_thinking=False，本地 golden 与
   upstream server 的 accept length 在 n=16/64 均吻合。
3. **加速比（真实 rollout 口径，temp=1.0）**：gsm8k c=1 **2.99x**、c=16 2.06x；perfectblend c=1
   2.46x、c=16 1.91x。

**三条踩坑沉淀（贯穿全文）**：
- **评测口径必须对齐 evaluate 金标准**：temp=1.0（走 rejection sampling，**非 greedy**）、
  max_new_tokens=2048、seed=980406、enable_thinking=False、chat template、纯 softmax multinomial
  （无 top_p/top_k/min_p 截断）。详见 §2.4。
- **accept length 差异可由 Python env（数值后端）造成**：同代码同 ckpt，主 env 得 6.07、sglang env
  得 6.22——bf16 GEMM/attention 在 near-tie logit 上的浮点差异被 temp=1.0 采样放大。**对拍必须同 env。**
  详见 §4.4。
- **起 server 用 flashinfer attention backend**（target + draft）：triton backend 在 spec-decode
  长序列 GEMM 会 `CUBLAS_STATUS_EXECUTION_FAILED` 崩。详见 §4.5。

---

# 第一部分 · 相关项目背景理解

## 1. 两侧 draft 架构 + upstream 原生 DSPARK（为什么可集成）

### 1.1 参考实现坐标（只读）

| 项目 | 路径 | 说明 |
|---|---|---|
| upstream sglang（**当前基座**）| `third_party/sglang`（submodule → fork `happyfanta00/sglang` 分支 `dspark-opd-patches`，base HEAD `7e7129a`）| 我们的 T1/T2 改动 commit `5a309fe` |
| fork sglang-dflash（**已放弃**）| 曾 vendored 于 `third_party/sglang-dflash/` | 已 `git rm`；如需查阅见 git tag `backup/pre-fork-cleanup`。下文凡引 `third_party/sglang-dflash/...` 路径均指该历史快照 |
| verl 消费端参考 | `/home/ec2-user/efs_data/workspace/Draft-OPD/verl/` | Stage-M2 rollout 接线参照 |
| DSpark 模型/推理（本 repo）| `deepspec/modeling/dspark/`、`deepspec/eval/dspark/`、`third_party/verl/recipe/dspark_opd/` | 正确性金标准来源 |

### 1.2 DSpark 与 DFlash draft backbone 同构

DSpark 与 DFlash 的 draft backbone **同构**（都是并行块预测 + target-hidden cross-attention + 非因果
block 注意力 + 无 embedding/lm_head），差异只在**输出端的 markov head**。这是当初判断"可集成"的基础。

**DSpark draft 前向（本 repo）**：一个 anchor 一个 block：`[anchor_token, mask, mask, …]`（长
`block_size`），一次并行前向出整块 hidden，块内**非因果**注意力，K/V 同时来自 **target hidden
（context）** 与 **noise（draft query）** 两个流。

| 组件 | 位置 | 说明 |
|---|---|---|
| backbone 前向 | `deepspec/modeling/dspark/qwen3/modeling.py:361-386` `_forward_backbone` | 输入 `noise_embedding` + `target_hidden_states`；后者经 `fc`+`hidden_norm` 投影当 cross-attn K/V |
| cross-attention K/V 双流 | `modeling.py:103-112` `k_ctx=k_proj(target_hidden)`, `k_noise=k_proj(hidden)` 拼接 | 与 DFlash 一致，是复用其 KV 物化的技术基础 |
| 提议采样 | `modeling.py:309-333` `sample_draft_tokens` → `markov_head.sample_block_tokens` | 串行 markov 循环 |
| block/anchor 语义 | `modeling.py:452-466`、eval `draft_ops.py:106-139` | anchor 占 slot 0；DSpark slot i 预测 `anchor+1+i`（`label_offsets=arange(1,block_size+1)`）|

### 1.3 markov head（DSpark 与 DFlash 的唯一差异点）

`deepspec/modeling/dspark/markov_head.py`：低秩加性 logits 偏置，键是**上一个采样 token**（串行依赖）。

- `VanillaMarkov`（`:8-90`）：`bias = markov_w2(markov_w1[prev_token])`，memoryless（**本 ckpt 用这个**）。
- `GatedMarkovHead`（`:93-122`）：多 `gate_proj`，用 `sigmoid([hidden; prev_emb])` 门控，依赖 hidden。
- `RNNHead`（`:125-284`）：本项目不用；upstream `build_markov_head` 里对应分支也在。

### 1.4 DFlash vs DSpark 差异总表（含集成结论）

| | DFlash | DSpark | 集成结论 |
|---|---|---|---|
| backbone | 并行块 + target-hidden cross-attn | 同 | 复用 KV/attention/kernel |
| block 采样 | 一次并行 greedy | **markov 串行**（vanilla/gated）| upstream 原生 `sample_block()` |
| block 长度 | 固定 | 可解耦 verify 长度 | upstream `speculative_dspark_block_size`（verify=γ+1）|
| verify 采样 | target_only（丢 q）| **完整-draft-分布 rejection**（消费 q）| upstream `reject_sampling.py` |
| 权重 | 无 markov | 多 `markov_w1/w2`(+`gate_proj`) | upstream load_weights 前缀分流 |

### 1.5 upstream 原生 DSPARK：关键模块与证据坐标

base HEAD `7e7129a`（vendored 为 submodule `third_party/sglang`，见 §7.1）。fork（旧 sglang-dflash）里**不存在**的新增：

| 模块 | 位置 | 作用 |
|---|---|---|
| 纯 Triton classic rejection 内核 | `python/sglang/srt/speculative/reject_sampling.py:71` | `coin*q<p` ⇔ 接受概率 `min(1,p/q)`；`:109-156` 残差 `relu(p−q)`；全接受从纯 target 采 bonus。**只依赖 triton，无需重编 sgl_kernel** |
| 原生 DSpark 模型 | `python/sglang/srt/models/dspark.py:505,509` | `Qwen3DSparkModel`（`EntryClass`）+ `VanillaMarkov/GatedMarkovHead/RNNHead` + `markov_head.sample_block()` 出 `corrected_logits` |
| DSpark verify | `dspark_components/dspark_verify.py:653` `accept_draft_tokens` | 非 greedy 默认走 `AcceptSampling`→`chain_speculative_sampling_triton`（classic 内核），`:675` 对 `corrected_logits` 做 softmax 得真实 draft 概率 q |
| DSpark worker v2 | `dspark_components/dspark_worker_v2.py` | `DSparkWorkerV2`，同时驱动 overlap / 非 overlap（`spec_info.py:254-259`）|
| load_weights 前缀分流 | `models/dspark.py:38-67` | 按 `markov_head.`/`confidence_head.` 前缀分流；markov 权重名与本 repo `markov_head.py` 逐字一致 → 权重可直接加载 |
| block/verify 解耦 | `spec_info.py:222-224` + server arg `speculative_dspark_block_size` | draft worker 用 `num_draft_tokens-1`；verify window = γ+1（无 fork 的固定块长妥协）|

**注意**：upstream 里 **DFLASH 仍用旧的 target_only + draft_probs=0**（`dflash_utils.py:685,702`），
只有 **DSPARK** 拿到了新的 classic rejection 采样器。这是选 DSPARK 路径而非 DFLASH 的又一理由。

---

# 第二部分 · 我们的需求与可行性分析

## 2. 需求、正确性标准与方案选型

### 2.1 需求（用户目标）

把 DSpark-OPD 训练里"从 cache 读的 response"替换为"SGLang 现场投机解码产生的 response"，其余训练逻辑
与 tensor-contract 不变；同时把 rollout 过程中的 **draft 信息**导出供后续实验。核心约束是——投机解码
**必须正确（无损）**，且**关注 draft model 本身的采样过程**（这是本项目区别于普通投机解码加速的地方）。

### 2.2 正确性标准（用户澄清 —— 本项目的核心问题）

正确性 = 不仅 **target 分布无损**，还要 **draft model 的接收/采样过程完全无损**，即支持
**完整-draft-分布 rejection sampling**：

> draft 从自己的分布 q 采样、接受概率 `min(1, p/q)`、拒绝按残差 `(p−q)₊` 重采样。

本 repo `deepspec/eval/base_evaluator.py:252-285`（`sample_residual`）就是这个标准。**这才是本项目关注
draft model 的核心信号**——如果把 draft 当成确定性 argmax 点质量、抹掉 q，就丢掉了要研究的东西。

### 2.3 可行性分析：fork 不达标，upstream 原生达标

**（A）fork 为什么不满足**（有 CUDA 源码为证）：fork 唯一 verify 原语
`tree_speculative_sampling_target_only`
（`third_party/sglang-dflash/sgl-kernel/csrc/speculative/speculative_sampling.cuh:72-160`），
在 `threshold_single=threshold_acc=1.0`（fork 默认）下：

```
prob_acc += target_probs[draft_token]        // 只累加 TARGET 概率 p
if (coin <= prob_acc / threshold_acc || ...) accept
```

- **draft 概率 q 完全不参与接受判定**；调用方一律传 `draft_probs=0`。
- 拒绝残差 `relu(target − draft)` 因 `draft=0` 退化为"从 target 里排除已试 token"。
- `.cuh:91` 明写 `// FIXME: leverage draft probs`。
- **数学性质**：对 **target 输出分布无损**（EAGLE 用它也无损的原因）；**但不是完整-draft-分布
  rejection sampling**——把 draft 当确定性 argmax 点质量，`min(1,p/q)` 里的 q 被抹掉。对关注 draft
  model 的项目恰好抹掉要研究的信号。⇒ **fork 不达标。**

**（B）upstream 为什么达标 + 权重兼容**：见 §1.5。关键：
- upstream 新增纯 Triton `reject_sampling.py`（`coin*q<p` ⇔ `min(1,p/q)` + 残差 `(p−q)₊`），与本 repo
  `base_evaluator.py:252-285` 逐行等价；整套 `dspark_components/`（原生 markov head、`sample_block()`、
  `DSparkWorkerV2` 驱动 overlap/非 overlap）已就绪。
- upstream draft 架构类就叫 `Qwen3DSparkModel`，与本 repo ckpt 的 `architectures` 字段同名；
  `load_weights` 按 `markov_head.`/`confidence_head.` 前缀分流，markov 权重名（`markov_w1/w2/gate_proj/
  joint_proj`）与本 repo `markov_head.py` 逐字一致 → **同源共设计，权重可直接加载**。
- 且顺带解决了 fork 方案遗留的所有悬案（markov 串行采样、block/verify 长度解耦、overlap）。

- ⚠️ **硬前提**：upstream `build_markov_head`（`models/dspark.py:264-271`）对 `markov_rank<=0` 直接
  raise。早期候选 ckpt `deepseek-ai/dflash_qwen3_4b_block7` 是 `markov_rank=0`，**无法驱动 upstream 原生
  rejection 路径，已弃用**。使用的可用 ckpt 见 §7.1。

### 2.4 评测金标准（贯穿所有验证脚本）

**用户拍板：`eval.py` / `deepspec.eval` 的 evaluate 是评测金标准。任何评测代码（benchmark、probe、
加速比、接受率对拍）的采样/生成设置都必须与它对齐。** 金标准（以代码为准）：

| 设置 | 值 | 出处 |
|---|---|---|
| temperature | 1.0 | `eval.py:37` |
| max_new_tokens | 2048 | `eval.py:36` |
| seed | 980406 | `eval.py:70` |
| per-sample seed | `seed_all(seed + idx)`（idx=shuffle 后全局位置）| `base_evaluator.py:530` |
| enable_thinking | False | `base_evaluator.py:538` |
| 采样 | 纯 softmax multinomial，**无 top_p/top_k/min_p 截断** | `deepspec/utils/sampling.py:logits_to_probs` |
| chat template | `encode_chat_messages`, add_generation_prompt=True | `base_evaluator.py:534` |

- 已核实 sglang server 默认恰好也是 `top_p=1.0/top_k=-1/min_p=0.0`（无截断）；评测脚本已显式写出防漂移。
- **为什么必须 temp=1.0 而非 greedy**：temp=0(greedy) 走 argmax 匹配
  （`compute_dflash_accept_len_and_bonus`），**不进** rejection sampling verify 分支；temp=1.0 才走
  `min(1,p/q)` 完整-draft-分布 rejection，且与真实 rollout / evaluate 口径一致。
- **口径陷阱**：`spec_bench.py` 内部 `mean_accept_length` 变量是 `completion/verify − 1`（去 bonus），
  比本 repo evaluate 的 `completion/verify` 小 1，横向比时务必 +1 对齐。
- 各评测脚本默认已对齐：`spec_bench.py`（temp=1.0/max_tokens=2048/chat template）、
  `dspark_server_accept_probe.py`、`dspark_local_eval_accept.py`。

---

# 第三部分 · 代码开发方案（Stage-M0 / M1 / M2 / M3）

> 每阶段三段式：**① 具体做法 → ② smoke-test 测试方法与结果 → ③ 用户手动测试方法**。
> M0/M1 已完成（有实测结果）；M2/M3 待做（列方案与验证设计）。

## 3. Stage-M0 — upstream sglang 环境（✅ 完成）

### 3.1 具体做法

**基座迁移意味着依赖大改**：upstream HEAD 依赖激进（torch==2.11.0 / transformers==5.12.1 / CUDA cu13 /
sglang-kernel==0.4.4 / flashinfer 0.6.14），且原生 DSPARK 链路**只在 HEAD、未进任何 release tag**
（纯 Triton `reject_sampling.py` 从 v0.5.14 进 release，但 worker/verify 没进）⇒ 只能用 HEAD 源码。

- **env**：`~/.venv/dspark-opd-sglang`（用户重建的全新 Python 3.11）。**主 env `~/.venv/dspark-opd`
  永久不动**（回退基线，transformers 5.10.2 / torch 2.9.1+cu128）。
- **本机满足 cu13**：toolkit `/opt/pytorch/cuda` 是 13.0，driver 13.2；`CUDA_HOME=/opt/pytorch/cuda`。
- **走最小子集路线**（flashinfer 全量非必需；DSPARK spec 链只需 torch/triton/msgspec）。两个坑：
  1. rust 扩展报错 → `export SGLANG_BUILD_RUST_EXTS=none`（`python/setup.py:46`）跳过，DSPARK 不需要。
  2. 逐步补 import 依赖（pybase64/IPython/gguf/openai-harmony 等）。
- **安装脚本**：`scripts/opd/setup_env_sglang_upstream.sh`（STEP 1 torch → STEP 2 sgl-kernel →
  STEP 3B 全量依赖优先 / 3A 最小子集兜底 → STEP 4 editable 装 sglang 本体 → STEP 5 冒烟 → STEP 6 冻结）。
  依赖冻结：`docs/opd/pip-freeze-sglang-upstream.txt`（168 包）。
- fork 的旧安装脚本 `setup_env_sglang.sh`、`pip-freeze-sglang.txt` 等已随 fork 路线一并删除。

### 3.2 smoke-test 方法与结果

**smoke-test = 安装脚本内置 STEP 5**：`import sglang` + 逐个 import 原生 DSPARK 模块，全部成功则通过。

```python
mods = [
    "sglang",
    "sglang.srt.speculative.reject_sampling",
    "sglang.srt.speculative.dspark_components.dspark_verify",
    "sglang.srt.speculative.dspark_components.kernels.dspark_accept",
    "sglang.srt.models.dspark",
]
# 逐个 __import__，全成功打印 "[Stage-M0] UPSTREAM DSPARK IMPORT OK"
```

**结果（✅）**：torch 2.11.0+cu130 + H100 可用；`import sglang` OK；原生 DSPARK 模块全部加载 OK
（`reject_sampling` / `dspark_components.dspark_verify` / `models.dspark.{Qwen3DSparkModel,build_markov_head}`）。

> 另有主 env（`dspark-opd`）自检脚本 `scripts/opd/check_env.py`（S0 smoke），验证 DeepSpec + verl
> 0.7.0 运行时子集可 import；末行 `[S0] ENV OK` 即通过。这是训练侧的环境自检，与 sglang env 独立。

### 3.3 用户手动测试方法

```bash
# 由用户手动逐 STEP 执行（Claude 不直接操作 python 环境）；任一 STEP 报错停下贴回 traceback
bash scripts/opd/setup_env_sglang_upstream.sh
# 观察 STEP 5 是否打印 "UPSTREAM DSPARK IMPORT OK"；STEP 6 冻结 pip-freeze-sglang-upstream.txt
```

---

## 4. Stage-M1 — 正确性 + 接受率 + 加速比验证（✅ 全部达成）

**ckpt（用户提供，可用）**：`/mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest`
（`markov_rank=256`、`markov_head_type=vanilla`、`block_size=7`、`enable_confidence_head=true`、
target=Qwen3-4B、hidden=2560）。

### 4.0 前置 smoke-test：离线权重键映射校验（无需 CUDA/server）

- **具体做法**：不实例化重模型，纯 replay upstream `load_weights` 的前缀分流 + stacked-params 融合规则，
  diff ckpt key 集与 upstream 期望参数集，报 missing / unexpected。脚本
  `dspark_upstream_weight_keymap_check.py`。
- **smoke 方法与结果**：
  ```bash
  python scripts/opd/dspark_upstream_weight_keymap_check.py \
      /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest
  # 期望末行：RESULT: CLEAN LOAD ✅
  ```
  结果 **47/47 CLEAN** → 权重可直接被 upstream `Qwen3DSparkModel` 加载。

### 4.1 验证点1 —— 正确性（内核级等价 + 无损，最强证据，无需 server）

- **具体做法**：`scripts/opd/dspark_rejection_equivalence_check.py`（upstream env + H100）。喂两条腿完全
  相同的 (p, q, 候选链, coins)：(A) upstream `reject_sampling.chain_speculative_sampling_triton`
  （DSPARK verify 实际调用）；(B) 本 repo `base_evaluator.py:252-285` 参考实现。两个互补测试：
  - **TEST 1（等价）**：相同输入下两采样器输出分布一致 ⇒ 同一个采样器（消费 q、`min(1,p/q)`、残差 `(p−q)₊`）。
  - **TEST 2（无损）**：候选从 q 重采样后 first-commit token 边缘分布收敛到 target p0 ⇒ q 被洗掉，精确
    保持 target 分布。

- **smoke 方法与结果**：
  ```bash
  # upstream env（~/.venv/dspark-opd-sglang）+ CUDA
  python scripts/opd/dspark_rejection_equivalence_check.py --trials 300000 --vocab 32 --gamma 7
  # 期望末行：RESULT: EQUIVALENT & LOSSLESS ✅
  ```

  | 配置 | TEST1 first-commit Δ | TEST1 accept-len Δ | TEST2 upstream vs p0 | TEST2 repo vs p0 |
  |---|---|---|---|---|
  | gamma=4, vocab=16, N=2e5 | 0.0026 | 0.0027 | 0.0008 | 0.0043 |
  | gamma=7, vocab=32, N=3e5 | 0.0041 | 0.0068 | 0.0011 | 0.0035 |

  （gamma=7 = 本 ckpt 真实 block_size；全部 < 蒙特卡洛容差）→ **"draft 接收/采样过程完全无损"在采样器
  层面数学坐实**。

### 4.2 验证点2 —— 接受率对拍（≤2% 吻合）

- **具体做法**：口径两边严格一致——golden `acceptance_length_sum/proposal_count`
  （`base_evaluator.py:482`）== server `Σcompletion_tokens/Σspec_verify_ct`（都含每步 bonus）。对齐条件：
  同 gsm8k、seed 980406、temp=1.0、enable_thinking=False。**server 与 golden 均在 sglang env**（同数值
  后端，见 §4.4 为何必须同 env）。两个脚本：
  - `dspark_local_eval_accept.py`：本地 eval.py golden 基准（薄 wrapper，禁 confidence/entropy recorder）。
  - `dspark_server_accept_probe.py`：server 侧 accept length（喂与 eval 完全相同的 prompt，拿 meta_info.spec_verify_ct）。

- **smoke 方法与结果**：
  ```bash
  # (a) 本地 golden（在 server 未占用的 GPU 上，如 GPU1）
  CUDA_VISIBLE_DEVICES=1 python scripts/opd/dspark_local_eval_accept.py \
      --target Qwen/Qwen3-4B --draft <ckpt> --task gsm8k --n 64 --temperature 1.0
  # (b) server 侧（DSPARK server 已起）
  python scripts/opd/dspark_server_accept_probe.py --task gsm8k --n 64 --temperature 1.0
  ```

  | n | 本地 eval.py golden（HF 前向）| upstream server（DSPARK, micro）| 差异 |
  |---|---|---|---|
  | 16 | 6.0735 | 5.9539 | 2.0% |
  | 64 | 6.2233 | 6.2906 | 1.1% |

  golden n=64 per-position accept rate `[0.937,0.874,0.807,0.745,0.692,0.636,0.581]`（逐位单调递减，合理）。

### 4.3 验证点3 —— 加速比（temp=1.0 真实 rollout 口径）

- **具体做法**：同引擎 A/B——DSPARK spec server vs 纯 target baseline server，同 upstream env、同
  flashinfer backend、同 cuda graph。脚本 `scripts/opd/spec_bench.py`（seed 980406、**temp=1.0**、
  max_new_tokens=256、chat template + enable_thinking=False）。accept len 列已换算成 evaluate 口径
  `completion/verify`（= 内部值 +1）。

- **smoke 方法与结果**：
  ```bash
  # 分别对 baseline server / dspark server 各跑一次，再 --compare
  python scripts/opd/spec_bench.py --port 30000 --n 128 --concurrency 16 --tag gsm8k_baseline_c16 \
      --dataset eval_datasets/gsm8k.jsonl --out docs/opd/bench/stageM1_gsm8k_baseline_c16.json
  python scripts/opd/spec_bench.py --port 30000 --n 128 --concurrency 16 --tag gsm8k_dspark_c16 \
      --dataset eval_datasets/gsm8k.jsonl --out docs/opd/bench/stageM1_gsm8k_dspark_c16.json
  python scripts/opd/spec_bench.py --compare \
      docs/opd/bench/stageM1_gsm8k_baseline_c16.json docs/opd/bench/stageM1_gsm8k_dspark_c16.json
  ```

  | 数据集 | 并发 | n | baseline tok/s | DSPARK tok/s | **加速比** | 延迟 p50 (s) | accept len |
  |---|---|---|---|---|---|---|---|
  | gsm8k | c=1（低并发）| 24 | 221.48 | 661.96 | **2.99x** | 1.154 → 0.356 | 6.162 |
  | gsm8k | c=16（高并发）| 128 | 2675.06 | 5511.96 | **2.06x** | 1.448 → 0.631 | 6.212 |
  | perfectblend | c=1 | 24 | 220.96 | 542.87 | **2.46x** | 1.152 → 0.384 | 5.101 |
  | perfectblend | c=16 | 128 | 2626.80 | 5021.02 | **1.91x** | 1.448 → 0.629 | 5.535 |

  结果落盘：`docs/opd/bench/stageM1_{gsm8k,pb}_{baseline,dspark}_c{1,16}.json`。

- **读表注意（避免误读）**：
  - **加速比**随并发下降（低并发 > 高并发）：符合投机解码规律——高并发下 GPU 已算力饱和，投机边际收益缩水。
  - **accept len 与并发基本无关**：accept length 逐序列逐 token 决定，与 batch 里有多少并发请求无关。表中
    c=1 与 c=16 的 accept len 小幅差异来自两者样本量不同（c=1 测 n=24、c=16 测 n=128），是不同样本子集的
    难度分布差异；方向上高并发反而**略高**（若并发会拉低 accept，应相反）。两数据集 accept len 不同
    （gsm8k ~6.2 vs pb ~5.x）则是数据集本身差异。
  - gsm8k accept len 6.16/6.21 与 §4.2 接受率对拍 golden 6.22 / server 6.29 同口径几乎相等，互为交叉验证。

### 4.4 ⚠️ 方法论：accept length 差异可由 Python env（数值后端）造成

**同一份 eval.py 代码、同一个 ckpt（`step_latest`）、同一批 64 条样本（md5 相同）、单 GPU 与 8 GPU 结果
相同**，唯一变量是运行 env，accept length 差 ~2.4%：

| | eval.sh（主 env `dspark-opd`）| 对拍（sglang env）|
|---|---|---|
| **accept_len** | **6.07** | **6.22** |
| transformers / torch / CUDA | 5.10.2 / 2.9.1+cu128 | 5.12.1 / 2.11.0+cu130 |
| proposal_count | 3491 | 3224 |

**根因**：不同 transformers + torch/CUDA 后端的 bf16 attention/GEMM 在 near-tie logit 上给出微小浮点差异，
temp=1.0 采样对其极敏感（一个 token 分叉 → 整条轨迹发散，proposal_count 也随之不同）。这是引擎级现象、
不是 bug（与投机解码本身无关）。

**排查中逐一实测排除的因素**（都不是差异来源）：样本集 md5 相同；temp/seed/per-sample seed(`seed+idx`)；
enable_thinking=False；confidence_threshold=0.0（`_confident_prefix_length` 在 ≤0 时 return block_size，
不截断）；entropy-stats（只在 `_post_verify` 记录，不改前向）；max_new_tokens（1024=2048=6.2233，这 64
条无超 1024 者）；GPU 数（单 GPU=8 GPU=6.22，n=4 时逐位相同 6.38）。

**教训**：**accept length / 无损性对拍必须在同一 env（同数值后端）进行才公平。** §4.2 的 server 与 golden
恰好都在 sglang env，是对的；若拿 sglang server 去比主 env 的 6.07 会引入 env 系统性偏差。

### 4.5 用户手动测试方法：起 server 的关键事实

- **必须用 flashinfer attention backend**（target + draft）：triton backend 在 spec-decode 长序列 GEMM
  会 `CUBLAS_STATUS_EXECUTION_FAILED` 崩（cuda graph capture 时 draft `compute_base_logits` 的 bf16
  GEMM；纯 eager 下长序列 decode 也崩）。flashinfer 下长序列 decode + cuda graph capture 都稳。
- gamma 自动从 ckpt config 读（block_size=7 → num_draft_tokens=8），无需手动传。
- 启动脚本 `scripts/opd/launch_sglang_upstream_dspark.sh {dspark|baseline}`（长时进程，用户手动执行）：
  ```bash
  # DSPARK 投机解码 server（默认 flashinfer，gamma 自动读）
  bash scripts/opd/launch_sglang_upstream_dspark.sh dspark
  # 纯 target baseline server（加速比对照基准）
  bash scripts/opd/launch_sglang_upstream_dspark.sh baseline
  # 纯 eager 定位问题：DSPARK draft 只认总开关 disable_cuda_graph，disable_decode_cuda_graph 不管用
  NO_CUDA_GRAPH=1 bash scripts/opd/launch_sglang_upstream_dspark.sh dspark
  ```
  脚本已自包含 CUDA env 前缀（`CUDA_HOME=/opt/pytorch/cuda` + PATH/LD_LIBRARY_PATH）。

---

## 5. Stage-M2 — verl rollout 接 upstream DSPARK sglang（⏳ 待做，已出方案）

**目标**：把 DSpark-OPD 训练里"现场由 draft nn.Module block 采样出的 response"替换为"upstream DSPARK
sglang 现场投机解码产生的 response"，其余训练逻辑不变、tensor-contract 不变（`docs/opd/tensor-contract.md`）；
同时用**最小契约**导出 draft 信息。

### 5.0 接入拓扑与起点（对齐 verl 标准 async sglang rollout）

> **本节先厘清"标准做法"与"当前起点"，是 §5.1 起全部子任务的前提。已核对 verl 本体、Draft-OPD 参考、
> 当前 dspark_opd recipe 三处源码。**

**verl 的标准 sglang rollout 抽象**（`verl/workers/rollout/base.py:28,80,88`）：`BaseRollout` 基类 +
`_ROLLOUT_REGISTRY` + `get_rollout_class(name, mode)`。sglang 有两条官方路径：
- **sync（进程内 Engine，默认/主流）**：`SGLangRollout` 在 worker 进程内嵌 `sglang.Engine` 子类，
  token-in-token-out，不走 HTTP（`sglang_rollout.py:134,257,484`）。
- **async（server 模式）**：`ServerAdapter(BaseRollout)` + `SGLangReplica(RolloutReplica)` + Ray actor
  `SGLangHttpServer`（`async_sglang_server.py:51`、`replica.py:231-239`），控制面走 HTTP、生成走进程内
  `tokenizer_manager.generate_request`。
- ⚠️ **verl 本体对投机解码无一等公民支持**（全树 grep `speculative/eagle/draft_model` 零命中），只能靠
  `engine_kwargs.sglang` 透传底层 sglang 的 spec-decode 参数。

**决策（用户已定）：接入走 async server 模式，对齐 Draft-OPD 参考实现**
（`Draft-OPD/verl/.../sglang_rollout/`）：
- 用 verl 标准 `SGLangReplica` / `ServerAdapter` 起 sglang，生成走 `SGLangHttpServer.generate` →
  `tokenizer_manager.generate_request`（`async_sglang_server.py:208,464-642`），**不自建独立 `/generate`
  HTTP 客户端**（参考实现里裸 `/generate` 端点尚是 TODO，`:474`）。
- draft 信息复用 sglang 的 **`customized_info → meta_info`** 通道（§5.1），verl 侧在 `generate()` 里从
  `meta_info` 抽字段塞进 `extra_fields`（仿 `async_sglang_server.py:589-627`），首 token 偏移仿
  `align_dflash_reject_token_mask`（`utils.py:29-53`）。
- **与参考的唯一实质差异**：参考回传"大集"（reject_mask + 被拒 draft 四元组 anchor/offset/token_id/
  teacher_logp）；我们回传**最小集**——**单条 per-token `dspark_accept_state` 流**（accept/reject 二态），
  anchor/logp/q 训练侧凭它重 forward（§5.1、§5.11）。
- **基座差异需对齐**：参考依赖 fork `sglang-dflash` 才吐出这些 `meta_info` 字段；我们改基座 upstream，需
  **在 upstream sglang（submodule `third_party/sglang`）里自己生产这条流**（§5.4 生产端实现，已完成）——这是与参考最大的工作量差异。

**⚠️ 当前起点（务必先知）**：当前 dspark_opd recipe **默认并不走 sglang，也不经 verl 的
`generate_sequences` 入口**。默认路径是融合单 RPC `DSparkActorRolloutRefWorker.train_step`
（`worker.py:344`），在一个 RPC 内做 rollout→teacher scoring→update，rollout 段直接对 attach 的 draft
nn.Module 做 block-parallel 前向采样（`block_rollout.py:32,92`）；`DSparkRollout(BaseRollout)`
（`rollout.py:25`）目前只是 registry 合规占位（仅旧 3-RPC 路径用到）。cache 只提供输入特征
（`target_hidden_states`），非读"现成 response"。teacher（target）是 **co-resident** 建在 actor worker
进程内（`worker.py:152` `_build_teacher`），非独立 worker。

**架构取舍（用户已定，勿再动训练骨架）**：Stage-M2 的实质工作是把 rollout 段从"draft module 现场 block
采样"切到"upstream DSPARK sglang 现场投机解码"。**只对齐 rollout 侧的 sglang 接入这一件事——在融合
`train_step` 内注入 sglang server 调用当 response 来源，保留融合训练循环 + co-resident teacher 不变。**
- 依据：已核实 Draft-OPD 参考走 verl **标准多 RPC `RayPPOTrainer.fit()`（未重写）** + teacher 是**独立
  rollout replica（独立 GPU 池 + 独立 sglang server，`is_teacher_model=True`）**。**完全对齐它 = 连带把融合
  `train_step` 拆回标准多 RPC + 把 co-resident teacher 拆成独立 replica——两处大重构，远超"换 response 来源"
  的目标，风险高。** 故 M2 不走这条。
- **可复用**：verl 的 rollout replica **抽象类**（`get_rollout_replica_class` + `RolloutReplica` +
  `ServerAdapter`）是独立标准件，可直接在融合 `train_step` 内当"起 sglang + 发 generate"的句柄复用，**不强制
  回到标准多 RPC fit**。T3（§5.6）即按此做。

**GPU 资源布局（用户已定）：HYBRID 共卡全 8 卡，NO_SHARD 保留、不 offload，rollout↔training 靠 sleep/wake
错峰。**

- **模式**：sglang rollout server 与 actor FSDP 训练 **colocate 在同一批 8 卡上时分复用**（verl 标准
  HYBRID，Draft-OPD 在其 `global_pool` 上即此法）。**不**走"rollout 独占卡"（那会引入阶段性闲置且不必要），
  **不**走异步 off-policy（§已评估放弃：我们 OPD 是严格 on-policy，`old_log_prob=log_prob.detach()` 使
  `ratio≡1`，异步 staleness 会破坏该近似，属改算法而非改部署）。
- **显存账（H100 80GB / bf16，粗算，未实测）**：draft 仅 **~1.5B**（embed/lm_head 冻结，优化器状态更少），
  teacher 4B 冻结（无梯度/优化器）。
  - **训练态**（sglang 睡、KV 释放）：draft NO_SHARD footprint（权重~3G + 梯度~3G + Adam fp32 三件套~18G）
    ~24GB + co-resident teacher 8GB ≈ **~32GB/卡**——与今天单卡训练态同，已验证能跑。
  - **rollout 态**：额外 sglang（`mem_fraction_static=0.4`，target 8G + draft 3G + KV + cuda graph）
    ≈ **~32GB**。
  - **共卡合计 ≈ 64GB < 80GB** ✓ → **两态可同时驻留，无需 offload，NO_SHARD 不用改。**
- **为什么 NO_SHARD 保留**：draft 只 1.5B，共卡放得下，故那个"为多卡梯度同步正确性刻意选 NO_SHARD"的决策
  （`loss-design.md §2.6.5`、memory `dspark-opd-multigpu-grad-sync-bug`）**不用动**，风险归零。
- **sleep/wake 仍保留**（即便显存够）：rollout 阶段让 sglang 拿更大 KV 池、训练阶段释放 KV 给激活，是 verl
  标准做法、近零成本。
- ⚠️ **安全阀（未来 OOM 再启用，现在不预防性优化）**：上面 16GB 余量靠"draft 1.5B + 我们序列短
  （resp=64/anchor=32，激活小）"。若未来长序列 / 大 batch 把 KV+激活吃满余量，**再考虑切 FULL_SHARD 分片或
  rollout 态 param/optimizer offload**——用户已定此为延后项。

### 5.1 具体做法（一）：从 sglang 取 draft 信息的最小契约设计

> **设计决策（用户拍板，2026-07-16 落地修正为单流）：不回传"被拒 token / draft 概率 q / teacher logp"那一
> 大堆。最本质、也最鲁棒的信息只有 **一条 per-token 流**——`dspark_accept_state`：输出序列中每个位置的
> accept/reject 二态（0=ACCEPT / 1=COMMIT_BOUNDARY）。其余（每轮 anchor 位置、draft 分布 q、student/teacher
> logp）训练侧凭这条流 + 重 forward 即可恢复。**
> 这把 sglang→verl 的契约压到最小，避免 fork 那套 5 字段元数据流的脆弱对齐、folded/greedy 路径 q 缺失、
> reject 内核改造等一系列坑（附录列出被绕过的坑）。
>
> **为何从"两条流"收敛到"一条流"（关键工程事实）**：原设计另回传第二条 per-round 的 `dspark_block_anchors`
> （每轮一个 anchor，长度 = `spec_verify_ct`）。但 `customized_info` 传输层**全程按 output-token 偏移切片/累积**
> （`output_streamer.py:545-549` 用 `req_values[send_token_offset:len(output_ids_)]`，且非流式请求也被
> `DEFAULT_FORCE_STREAM_INTERVAL=50`（`environ.py:402`）强制每 50 token 分块）——per-token 的 accept_state 完美
> 存活，per-round 的 anchor 列表会被 token 偏移**切碎/丢轮**。而 anchor 100% 可从 accept_state 推出（见下），
> 是纯冗余且恰是不可传输的那条 ⇒ **弃第二条流，只发 accept_state**。

**为什么一条流就够（信息完备性论证）**：`dspark_accept_state` 是 **per-output-token 三态**编码——
`0=ACCEPT`（block 内被接受的 draft）/ `1=COMMIT_BOUNDARY`（decode 轮末位 bonus/修正 token）/
`2=PREFILL_SEED`（prefill 产的 response 首 token，即 block-0 的 anchor）。boundary 与 seed 即分轮标记，训练侧
无损重建全部所需：
- **anchor / block 划分**：`len(anchors) == spec_verify_ct`；round 0 由 **seed（index 0）** anchor，其后每个
  decode round 由 **上一轮的 boundary token** anchor；block 内 accept 段长度 = 前导连续 `ACCEPT`(0) 数。
  （全接受轮和 reject 轮**都有 boundary**，修掉 fork "全接受轮不报 anchor" 缺口。）
- **accept 流**：从 anchor 起、到本 block boundary 前一位为止的 on-trajectory 段——重 forward draft 拿
  student logp（DSpark 前向不吃块内被拒 token_id，喂 `mask_token`，token_id 只在输出端当 label）。
- **reject 流**：boundary 位置的 token 就在 output 序列里（它是该 block 的 bonus / 修正 token，已在 response
  中），其 anchor 已知 → 重 forward 即可拿该位置的 draft 分布 q 和 student/teacher logp。**无需 sglang
  单独回传被拒 token id / q / teacher logp。**
- **reject 与纯 bonus 的区分不入流，由训练侧推**：block 长度 `< draft_token_num`（未提满）⇒ 该轮有 reject；
  `== draft_token_num`（提满）⇒ 纯 bonus 无 reject。

**⚠️ 为什么必须有 SEED（prefill 首 token）—— 血泪教训（金标准口径实测暴露）**：spec decode 的 response 首
token 由 **prefill**（`process_batch_result_prefill:265 req.output_ids.append`）产生、**不经**
`_resolve_spec_v2_tokens`。若流只覆盖 decode 轮，则流是 **decode 坐标**（长 D），而传输层
（`output_streamer.py:545-549`）用 **output_ids 坐标**（长 D+1）切片 `req_values[send_token_offset:len(output_ids_)]`
——**错位 1 位**。**无截断时** Python 切片 clamp 掉此错位、无害（`len==completion−1` "看似正确"）；**一旦 stop/EOS
截断**某轮中途，`output_ids_through_stop` 的裁剪与流错位，把尾部 boundary 切掉 → `#boundary < spec_verify_ct`、
丢轮（n=8 gsm8k 金标准口径实测 5/8 样本 FAIL）。**补 seed 后流成为真正的 per-output-token 对齐**：传输切片和
stop-trim 对流的裁剪与 output_ids **逐位一致**，`len(dspark_accept_state)` 精确 == `completion_tokens`。

**这条流在 sglang 的确切来源**（已核实源码，HEAD `7e7129a`）——**两个落点**，都在
`scheduler_components/batch_result_processor.py`：

| 落点 | 采什么 | 怎么采 |
|---|---|---|
| `process_batch_result_prefill`（首 token）| 1 个 `SEED` | `len(req.output_ids)==1 and is_dspark()` 时 append `2` |
| `_resolve_spec_v2_tokens`（每 decode 轮）| 本轮 chunk | `dspark_round_accept_state(num_accept_tokens-1)` = `[0]*(n-1)+[1]`，`extend` |

> ⚠️ **时序/对齐已核准 + 实测坐实**：`_resolve_spec_v2_tokens` 在 `process_batch_result_decode` 里 **早于
> `req.output_ids.extend()`** 运行，故本轮 `extend` 进 `customized_info` 的 per-token chunk 与随后 append 的
> output_ids 逐位对齐；prefill seed 同理占 output_ids[0]。用 grammar 截断后的 `len(accept_tokens)` 而非
> `num_correct_drafts_per_req_cpu[i]`，保证长度不变式。`num_accept_tokens==0`（grammar 立即 abort）跳过不记。
> seed **gated on `len(req.output_ids)==1`**：retracted-resume 会重跑 prefill 且 `customized_info` 不重置，
> 守卫防重复 seed（我们短序列 rollout 配置不触发 retraction；万一触发则长度校验响亮失败而非静默错位）。
> anchor **不单采**，由 seed + boundary 推出。

**回传通道（现成、零接线，与 fork 同型）**：`customized_info` 是一条已打通的通用 per-token 通道：
1. **生产**：`req.customized_info: Dict[str, List]`（`schedule_batch.py:951`）；DSPARK spec 每步提交多个 token，
   通用 collector（`batch_result_processor.py:159` `_maybe_collect_customized_info`，每 step 一元素）语义不适配，
   故在 `_resolve_spec_v2_tokens` 里直接 `extend`。
2. **打包**：`output_streamer.py:537-553` 已通用收集 `req.customized_info` 所有 key（按 token 偏移切片）→
   `BatchTokenIDOutput.customized_info`。**per-token 单流无需改打包。**
3. **进 meta_info**：`tokenizer_manager.py:_handle_batch_output`（`:1973-1978`）解包后累积 → `meta_info[k]`。
   **白名单无需登记**：非流式 `/generate` 路径无条件透出；流式白名单也是 `state.customized_info_accumulated.
   keys()` 自动发现（`:1511/2030/2076`），非静态表。

⇒ 只需在生产端写 `req.customized_info` **一个 key**（`dspark_accept_state`），即端到端透出到 `/generate`。
**无需碰 verify 内核、无需碰 reject_sampling、无需解除 `return_logprob` 限制、无需改打包/白名单**（logp 训练侧重算）。

### 5.2 具体做法（二）：draft 权重同步设计（Stage-M2 第二块 upstream 改动，调查结论）

> **一旦引入 sglang 引擎，RL 训练每个 step 更新完 draft 权重后，必须把新 draft 权重同步回 sglang 引擎里的
> draft worker，否则 rollout 永远用旧 draft。** 当前 recipe 不走 sglang（in-process FSDP module 生成，权重
> 天然可见），所以现在**没有也不需要**这一步——它是 M2 引入 sglang 后第一次出现的需求。已核对 verl 本体、
> upstream sglang、Draft-OPD 参考三处源码。

**verl 标准权重同步机制（通用，不分 draft/target）**：每 PPO step 在 actor 更新后，hybrid-engine 切 rollout
上下文 → 取 `actor_module_fsdp.state_dict()`、DTensor `.full_tensor()` all-gather（`fsdp_workers.py:674,706`）
→ `rollout.update_weights(per_tensor_param)`（`base.py:51`，**扁平 `(name,tensor)` 流，无 draft/target 概念**）
→ 底层 **CUDA IPC handle + tensor bucket**（非 NCCL、非 HTTP 传权重体；`weight_sync/utils.py:40-103`）→
`engine.update_weights_from_tensor`。async 模式走 HTTP `/update_weights_from_tensor`，但 **HTTP 只传 base64
的 IPC meta，权重本体从 GPU 直接拷**（`http_server_engine.py:351`）。触发点 `ray_trainer.py:2058`（每 step）。

**upstream sglang 侧现状（🔴 关键缺口）**：
- sglang **有** draft/target 分流骨架：`weight_updater.py:151-164` 按 `disable_draft_model` 标志路由到
  `tp_worker`（target）或 `draft_worker`；`io_struct.py:1605` 有 `disable_draft_model` 字段。
- **但 `DSparkWorkerV2` 根本没定义 `update_weights_from_tensor`**——请求经 `__getattr__`
  （`dspark_worker_v2.py:270-273`）**回落到 target worker**，结果**只更新 target、draft 纹丝不动**。
- 且 upstream 只有反向的 `disable_draft_model`（只能"只更 target"），**没有 fork sglang-dflash 的
  `draft_model_only`（"只更 draft"）**（`sglang-dflash io_struct.py:1356-1358`，upstream 无）。
- **底层机器齐全**：DSPARK 的 `draft_model_runner` 是独立实例、自带 `weight_updater`
  （`dspark_worker_v2.py:93-106`、`model_runner.py:378`）——**缺的纯粹是"把 tensor 请求接到 draft runner
  上"的入口方法 + 只更 draft 的路由参数**。

**Draft-OPD 参考怎么做到（fork sglang-dflash）**：verl 侧 `get_per_tensor_param(trainable_only=True)` 只导
trainable + `_map_opd_draft_weight_name` 剥 `draft_model.` 前缀（`sglang_rollout.py:215,238`）；透传
`draft_model_only=True` → Scheduler 选 `draft_worker` → EAGLE/DFLASH worker **先刷 draft_runner、靠
`draft_model_only` 短路跳过 target**（`eagle_worker.py:1018`）。**target 冻结、永不回灌。**

⇒ **本项目要在 upstream 复刻这条链路**：既要在 upstream sglang 补 draft-only 更新入口（任务 T2），又要在
verl 侧接线权重导出（任务 T6）。这是 §5.1 信息导出之外、Stage-M2 的**第二块并列 upstream 改动**。

### 5.3 任务拆分总览（6 个子任务 + 集成测试）

Stage-M2 改动横跨 upstream sglang 与 verl 两侧，拆成 6 个可**独立开发、独立测试**的子任务，最后一次集成。
`[U]`=改 upstream sglang，`[V]`=改 verl。

| 任务 | 侧 | 内容 | 依赖 | 详见 |
|---|---|---|---|---|
| **T1** | `[U]` | draft 信息导出（单条 per-token `dspark_accept_state` 流 → `customized_info` → meta_info；anchor 训练侧推）✅ | — | §5.4 |
| **T2** | `[U]` | DSPARK draft-only 权重更新入口（补 `update_weights_from_tensor` + `draft_model_only` 标志）✅ | — | §5.5 |
| **T3** | `[V]` | rollout 接入（HYBRID 共卡全 8 卡 + 融合 `train_step` 内 wake→`generate_request`→sleep；NO_SHARD 不 offload，不拆 fit、不拆 teacher）| — | §5.6 |
| **T4** | `[V]` | draft 信息传输（meta_info → `extra_fields`，首 token 偏移对齐）| T1, T3 | §5.7 |
| **T5** | `[V]` | draft 信息消费（重建 block plan + 重 forward 取 logp，删 fork 反推）| T4 | §5.8 |
| **T6** | `[V]` | draft 权重同步接线（每 step 只导 draft trainable 权重、剥前缀、透传只更 draft 标志）| T2, T3 | §5.9 |
| **集成** | 两侧 | 完整 rollout→teacher→update→weight-sync 闭环 | T1–T6 | §5.10 |

**依赖图**（可并行的两条线）：
```
[U] T1 (信息导出) ─────────────┐
[U] T2 (draft 权重入口) ──┐    │
[V] T3 (rollout 接入) ────┼────┼──→ T4 (传输) ──→ T5 (消费) ──┐
                          └────┼─────────────────→ T6 (权重同步)┼──→ 集成 §5.10
                               └───────────────────────────────┘
```
- **T1/T2/T3 无相互依赖，可三线并行起步**（T1、T2 是 sglang 源码改动，T3 是 verl 接线）。
- T4 需要 T1（有流可抽）+ T3（有 server 可打）；T5 接 T4；T6 需要 T2（有 draft 入口）+ T3（有引擎）。
- 建议顺序：先 T1+T2+T3 并行 → 再 T4→T5 与 T6 并行 → 最后集成。

> **每个任务下给出：① 做法（精确到函数）② 单独测试方案（脱离其它任务即可验证）。** 跨任务的端到端在
> §5.10 集成测试统一验。

### 5.4 任务 T1 `[U]` — draft 信息导出（单条 per-token 流）✅ 已实现

> **落地修正（见 §5.1）：只发一条 per-token 流 `dspark_accept_state`，anchor 训练侧从 boundary 推导。**
> 原"两条流"里的 per-round `dspark_block_anchors` 因 customized_info 传输层按 token 偏移切片而不可靠，且是纯
> 冗余，已弃。

**做法**（全部改在 upstream sglang（submodule `third_party/sglang`），base HEAD `7e7129a`。原则：**加一条流、不改现有逻辑**，逻辑抽成纯
函数便于 CPU 单测。改动仅 `batch_result_processor.py` 一文件 +63 行，additive）：

- **改点 1 — 常量 + 纯函数（新增，可孤立单测）**：在 `scheduler_components/batch_result_processor.py` 顶部加
  ```python
  DSPARK_ACCEPT_STATE_KEY = "dspark_accept_state"
  DSPARK_STATE_ACCEPT = 0            # block 内被接受的 draft
  DSPARK_STATE_COMMIT_BOUNDARY = 1   # decode 轮末位 bonus/修正 token
  DSPARK_STATE_PREFILL_SEED = 2      # prefill 产的 response 首 token（block-0 anchor）
  def dspark_round_accept_state(num_correct_drafts: int) -> List[int]:
      return [DSPARK_STATE_ACCEPT] * int(num_correct_drafts) + [DSPARK_STATE_COMMIT_BOUNDARY]
  ```
- **改点 2 — prefill 首 token seed（`process_batch_result_prefill`，`:265 req.output_ids.append` 之后）**：
  ```python
  if batch.spec_algorithm.is_dspark() and len(req.output_ids) == 1:
      if req.customized_info is None:
          req.customized_info = {}
      req.customized_info.setdefault(DSPARK_ACCEPT_STATE_KEY, []).append(DSPARK_STATE_PREFILL_SEED)
  ```
  - **必须有**：见 §5.1"为什么必须有 SEED"——否则流是 decode 坐标、与 output_ids 错位 1 位，stop-trim 时丢轮。
  - **gated on `len(req.output_ids)==1`**：防 retracted-resume 重跑 prefill 时重复 seed（守卫）。
- **改点 3 — 在 `_resolve_spec_v2_tokens` 的 `else`（非 retracted/finished）分支、
  `predict_tokens.append(accept_tokens)` 之前接线**（每 decode 轮）：
  ```python
  if batch.spec_algorithm.is_dspark() and num_accept_tokens > 0:
      state_chunk = dspark_round_accept_state(num_accept_tokens - 1)  # grammar 截断后的实际值
      if req.customized_info is None:
          req.customized_info = {}
      req.customized_info.setdefault(DSPARK_ACCEPT_STATE_KEY, []).extend(state_chunk)
  ```
  - **DSPARK 门控**：`batch.spec_algorithm.is_dspark()`（`spec_info.py:115`），不影响 EAGLE/DFLASH。
  - **grammar 截断修正**：用 `num_accept_tokens = len(accept_tokens)`（`_accept_grammar_tokens` 截断后）而非
    `num_correct_drafts_per_req_cpu[i]`，保证长度不变式。
  - **`num_accept_tokens==0` 跳过**：grammar 立即 abort、无提交的退化轮不记（否则 `n-1=-1` 会误吐 `[1]`）。
  - **retracted/finished 分支不记**：无提交，跳过，保持与 output_ids 对齐。
  - **时序对齐**：本函数早于 `:708 req.output_ids.extend()` 运行，chunk 与随后 append 的 output_ids 逐位对齐。
- **改点 4 — 打包不用改**：`output_streamer.py:537-553` 按 token 偏移通用收集 `req.customized_info`，per-token
  三态流完美存活（seed 让它 output_ids 逐位对齐，切片/裁剪与 output_ids 一致，见 §5.1）。
- **改点 5 — 白名单不用改**：非流式 `/generate` 无条件透 `meta_info[k]`（`tokenizer_manager.py:1973-1978`）；
  流式白名单也是 `state.customized_info_accumulated.keys()` 自动发现，非静态表。

**T1 单独测试方案**：
- **CPU 单测（无需 GPU/server）✅ 通过**——`scripts/opd/dspark_stream_unit.py`（import upstream 真符号，防漂移）：
  ```bash
  ~/.venv/dspark-opd-sglang/bin/python scripts/opd/dspark_stream_unit.py   # 期望末行 RESULT: T1 STREAM UNIT OK
  ```
  1. `dspark_round_accept_state(k)` == `[0]*k+[1]`；边界 k=0→`[1]`；SEED(2) 与 ACCEPT/BOUNDARY 三态互异。
  2. 从 SEED+accept_state 重建 anchors/block plan 与手算一致（seed anchor round 0，boundary anchor 下一轮）。
  3. **untrimmed 自洽**：`len == completion_tokens`（含 seed，精确）；`#(boundary) == spec_verify_ct`；重建 rounds == vct。
  4. **stop-trimmed 尾部**（复现金标准 bug 场景）：EOS 落某轮中途 ⇒ `#(boundary) == vct−1` 且流尾为非 boundary
     残段；重建仍恢复全部 vct 轮（末轮 partial）；`len == completion_tokens` 仍精确。
  5. 长度不变式：seed + trim 各组合下 `len == 实际 emit token 数`。
- **手动 server 验证（GPU，⏳ 加 seed 后待重测）**——**必须走评测金标准口径**（seed=980406 / chat template /
  `enable_thinking=False` / temp=1.0 / max_new_tokens=2048 / 无 top_p/top_k/min_p 截断，§2.4）。**复用现成
  对齐脚本 `dspark_server_accept_probe.py` 的 `--check-dspark-stream` 开关，勿手搓裸 token curl**（裸 token/开
  thinking 让生成分布跑偏，accept_len 掉到 ~2.9 而非 gsm8k 金标准 ~6.2，数值不可信）：
  ```bash
  bash scripts/opd/launch_sglang_upstream_dspark.sh dspark
  ~/.venv/dspark-opd-sglang/bin/python scripts/opd/dspark_server_accept_probe.py \
      --task gsm8k --n 8 --temperature 1.0 --check-dspark-stream
  # 期望末行：RESULT: T1 SERVER STREAM OK（逐样本断言 + 末尾 ok/fail/missing 汇总）
  ```
  - **断言（脚本内置，seed 版）**：`len(accept_state) == completion_tokens`（**精确**，seed 已含首 token）；
    `seed 数==1 且在 index 0`；`#(state==1) == spec_verify_ct`（或 `== vct−1` 且流尾非 boundary = stop-trim 残段）；
    重建 accept_len `completion/verify` == `spec_accept_length`。
  - **踩坑教训**：加 seed 前，n=8 gsm8k 金标准口径 5/8 样本 FAIL（`#boundary` 比 vct 少 1~2，集中在被截断样本）
    ——正是 decode/output_ids 坐标错位被 stop-trim 放大所致（§5.1）。这个 bug 在非金标准占位 prompt（accept_len
    2.9）下被掩盖，**是"评测必须对齐金标准"的活教材**。

### 5.5 任务 T2 `[U]` — DSPARK draft-only 权重更新入口 ✅ 已实现

**做法**（改 upstream sglang（submodule `third_party/sglang`），补齐 §5.2 缺口；4 文件 additive）：
- **补入口**（核心）：在 `DSparkWorkerV2` 新增 `update_weights_from_tensor(recv_req)`
  （`dspark_worker_v2.py`，仿 `eagle_worker_v2.py:1734`）。**先刷 `self.draft_model_runner.weight_updater`；
  若 `recv_req.draft_model_only` 为真则短路 return、不碰 target**（否则再刷 target，兼容 EAGLE 式"draft+target
  同刷"语义）。关键：定义成真方法后**盖过 `__getattr__` 回落**（§5.2 缺口的根因就是没这方法、回落到 target）。
- **补"只更 draft"路由参数**：新增 `io_struct.UpdateWeightsFromTensorReqInput.draft_model_only`（`:1607`，与
  既有 `disable_draft_model` 互为镜像）。**scheduler 路由 `weight_updater.py:151-157` 无需改**——它在
  `disable_draft_model` 未置时本就路由到 `draft_worker`（=DSparkWorkerV2），`draft_model_only` 由新入口内部消费。
- **打通传参链**：`draft_model_only` 从 `Engine.update_weights_from_tensor`（`engine.py:1067`）与
  `HttpServerEngineAdapter.update_weights_from_tensor`（`http_server_engine.py:78`，verl async 用这个）两条入口
  一路透传到 `UpdateWeightsFromTensorReqInput`。HTTP `/update_weights_from_tensor` 端点直接反序列化该 struct，
  字段自动带出。
- **⚠️ disk/ipc 路径未修（已知限制，tensor 路径不受影响）**：`base_spec_worker.py:99-115` 的
  `update_weights_from_{disk,ipc}` 遍历 `self.draft_worker.draft_runners`（复数 property，`base_spec_worker.py:25`
  返回 `[self.draft_runner]`），但 DSPARK 只有 `draft_model_runner`（无 `draft_runner` 单数）→ 走 disk/ipc 会
  `AttributeError`。**T6/HYBRID 只用 tensor（CUDA IPC）路径，完全绕开此问题**；若未来要 disk/ipc 再补。
- **键名对齐**：draft 权重 name（T6 剥掉 `draft_model.` 前缀后）经 `draft_model_runner.get_model().load_weights`
  加载，即 `Qwen3DSparkModel.load_weights`——与 §4.0 keymap 校验的 47/47 CLEAN 是**同一条加载路径**，键名兼容。

**T2 单独测试方案（✅ 已验证通过 2026-07-17）**（需 1 GPU，不依赖 verl）：`scripts/opd/dspark_draft_update_probe.py`。
**用 `HttpServerEngineAdapter` 起 server**（= verl async rollout 用的同一个类，直接走 CUDA-IPC 权重同步路径，即
T6 将用的路径）。验证走 **`/weights_checker`**（action=checksum，⚠️ 端点名是 `/weights_checker` 不是
`/check_weights`；返回体 `{success,message,ranks:[...],per_engine_checksum}`，逐参 checksum 在 `ranks[0]
["checksums"]`，draft 参数带 `draft.` 前缀，`weight_updater.py:60-69/269-298`）。

**单-ckpt 模式（更新通路 + round-trip）**：
```bash
CUDA_HOME=/opt/pytorch/cuda CUDA_VISIBLE_DEVICES=0 \
  ~/.venv/dspark-opd-sglang/bin/python scripts/opd/dspark_draft_update_probe.py \
    --target Qwen/Qwen3-4B --draft /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest
# 期望末行：RESULT: T2 DRAFT-ONLY UPDATE OK（adapter 自起 server 子进程，勿另起 launch 脚本）
```
读几个非 embed/lm_head draft 张量乘 1.05 扰动 → `update_weights_from_tensor(draft_model_only=True)` → 断言
（a）draft.* checksum 变、（b）target checksum 不变、（c）推回原值 round-trip 精确复原。

**两-ckpt 行为侧模式（`--draft-b`，主判据，✅ 实测通过）**：checksum 证"权重换了"，还要证"换权重带来 draft
预测行为的实质差异"，且**在 sglang 引擎内部、经 T2 真实更新通路观测**（对齐 verl/T6），非本地旁路。server 起
A → drive 固定 prompt → `update_weights_from_tensor(draft_model_only=True, flush_cache=True)` 推 B → 重 checksum
证 draft 变/target 冻结 → drive **同一 prompt**（flush 后强制重算首 block）。
```bash
CUDA_HOME=/opt/pytorch/cuda CUDA_VISIBLE_DEVICES=0 ~/.venv/dspark-opd-sglang/bin/python \
  scripts/opd/dspark_draft_update_probe.py --target Qwen/Qwen3-4B \
    --draft   /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest \
    --draft-b /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_3000   # 或 opd/step_16000
# 期望末行：RESULT: T2 TWO-CKPT UPDATE OK（draft 变、target 冻结）
```
- **为什么只看首 block**：投机解码无损 → 输出由 target 定，换 draft **不改输出、accept length 也可能看不出
  差异**；真正区分两 draft 的是 q 本身。首 block 的 anchor = prefill 首 token、输入完全确定，避免后续采样轨迹
  发散，是最干净的对比点。
- **实测结果（block7 base vs step_3000）**：(a) 45/46 draft 参数变、(b) 0 target 参数变、(c) 首 block per-slot
  q 变化**结构化**：靠近 anchor 的 slot 0 q(A)=q(B)=1.0（几乎不变），越靠后 draft 越不确定、训练改动越大——
  slot 4 从 q=0.963 掉到 0.530（唯一连 argmax 也变的位置），mean|Δq|=0.088。与 §4.2 per-slot 接受率单调递减
  同一直觉（draft 确定性沿 block 深度衰减、可塑性随之增大）。
- **⚠️ q 的观测靠一次性临时探针，已删（验证后清理）**：`/generate` 对 DSPARK 不暴露 draft q（`return_logprob`
  报错，§5.11）。验证时在 `dspark_verify.py:accept_draft_tokens` 的 `draft_probs` 处临时加了 env 门控
  （`SGLANG_DSPARK_PROBE_Q` / `SGLANG_DSPARK_PROBE_FILE`）的探针，把 proposed token 上的 q 落盘供 driver 读取+
  并排可视化。**T2 通过后已删除**（sglang 侧 `dspark_verify.py` 回到 pristine）。**如需复现 q 对比表**：按
  `dspark_draft_update_probe.py` 头部注释的代码片段把探针加回 `dspark_verify.py`（关键：gather q 用
  `draft_block.draft_tokens`、与 corrected_logits 对齐；**勿**用 `candidates`——slot 0 是 anchor、错位导致 q=0），
  driver 的 `--draft-b` 模式会自动 arm 探针并渲染表；无探针时 (c) 自动 SKIPPED、(a)(b) 仍为准。
- **踩坑沉淀（复现时注意）**：① `/weights_checker` 不是 `/check_weights`；② `HttpServerEngineAdapter` 无独立日志
  文件（server 是 multiprocessing.Process、stdout 混在父终端），故探针落盘到文件而非 scrape 日志；③ 同 prompt
  打两次第二次会命中 radix cache（`#cached-token`>0）不重算，须 `flush_cache=True`；④ 探针计数器进程级，跨
  drive 轮次靠"文件被 truncate 则重置"重新 arm；⑤ q gather 位置 off-by-one（candidates vs draft_tokens）会得 q=0。
- **辅助（本地、离线对比，非主判据）**：`dspark_draft_first_block_probe.py` 用本地 HF 前向对比两 ckpt 首 block
  的 per-slot sym-KL/argmax 分歧/q 移动——直接读加载的权重、跑得快，但**不经 sglang 通路**，仅作对照。

### 5.6 任务 T3 `[V]` — rollout 接入（HYBRID 共卡 + 融合 train_step 内切换生成）

**做法**（改 verl，按 §5.0 已定取舍：**HYBRID 共卡全 8 卡、融合 `train_step` 内注入 server + sleep/wake 错峰，
不拆 fit、不拆 co-resident teacher、NO_SHARD 不 offload**）：
- **复用 replica 类当句柄（HYBRID 模式）**：用 verl 标准 `get_rollout_replica_class("sglang")` /
  `SGLangReplica` / `ServerAdapter` + `init_hybrid`（`replica.py:146`，rollout 与训练 fused 同进程同卡）起
  upstream DSPARK sglang，spec-decode 参数经 `engine_kwargs.sglang` 透传（`speculative_algorithm=DSPARK` +
  draft ckpt + flashinfer backend + gamma 自动读）。**只把它当"起 server + 发 generate + sleep/wake"的句柄**，
  不接管训练循环。
- **注入点在融合 train_step，编排 sleep/wake**：把 `train_step`（`worker.py:344`）里 rollout 段的
  "draft nn.Module block 采样"（`block_rollout.py:32,92`）替换为
  `wake(sglang) → generate_request（`async_sglang_server.py:208`）→ sleep(sglang，释放 KV/权重备份 CPU)`；
  teacher scoring / update 两段保持不动（co-resident teacher 不改，NO_SHARD footprint 常驻）。
- **`DSparkRollout` 占位**：`generate_sequences` 仍不进主路径（融合 train_step 直接调 server），维持 registry
  合规即可。

**T3 单独测试方案**（不依赖 T1/T2 的字段与权重同步）：
- **显存实测（第一产出，最高优先）**：起 HYBRID replica，实测 **rollout 态峰值**（sglang wake + co-resident
  teacher + NO_SHARD 训练 footprint 同时在卡上）**≤ 80GB**——坐实 §5.0 那份"~64GB 粗算"，是整条路线成立的前提。
  若超，触发 §5.0 安全阀（切 FULL_SHARD / offload），**先测再往下做**。
- **最小 rollout spike**：验证"融合 `train_step` 进程内能干净 wake→`generate`→sleep DSPARK sglang replica"
  ——起 replica、发几个 gsm8k prompt（对齐评测金标准 temp=1.0/chat template），断言拿回 response 的
  形状/EOS/attention mask 合理、能被现有 tensor-contract 消费（形状/mask 一致），且 sleep 后训练态显存回落。
- **量级对拍**：同 prompt 下"sglang 产出"与旧"in-process block 采样产出"在长度分布/accept length 量级一致
  （temp=1.0 不要求逐字，参照 §4.4 env 敏感性——只比量级）。
- 复用 §4.3 `spec_bench.py` 口径确认接入后吞吐/accept 与 Stage-M1 server 一致（无回归）。

### 5.7 任务 T4 `[V]` — draft 信息传输（meta_info → extra_fields）

**做法**（改 verl，仿 Draft-OPD 现成循环）：在 `generate()` 里从 `meta_info` 抽**单个 key**
`dspark_accept_state` 塞进 `extra_fields`（仿 `async_sglang_server.py:589-627`）。**⚠️ 与原计划不同：T1 已在
sglang 侧用 `PREFILL_SEED` 补齐首 token（§5.1），流已 output_ids 逐位对齐、`len == completion_tokens`，故 T4
不再需要 `align_dflash_reject_token_mask` 式的"前补占位"**——直接对齐 response 坐标即可（若拿到的流仍是
`response_len−1`，说明连的是未加 seed 的旧 sglang，才需前补；正常路径不触发）。anchor 不传输——T5 消费时从
seed + boundary 推出（§5.1）。

**T4 单独测试方案**：
- **单测（mock，无需 server）**：构造带 `dspark_accept_state`（首元素 `SEED=2`）的假 `meta_info` dict → 跑抽取
  → 断言 `extra_fields` 该 key 存在、长度 == response_len、首元素为 seed、内容与输入一致。保留对"流长 =
  response_len−1"（旧 sglang 兼容）分支的前补占位覆盖。
- **接真 server（依赖 T1+T3 已就绪时做，属 T4 收尾）**：跑一个 prompt，断言 verl 侧 `extra_fields` 非空、
  长度 == response_len 自洽。

### 5.8 任务 T5 `[V]` — draft 信息消费（重建 block plan + 重 forward logp）

**做法**（改 verl）：从 `dspark_accept_state` 推出 anchors——round 0 由 **SEED（index 0）** anchor，其后每个
decode round 由 **上一轮 boundary token** anchor（尾部非 boundary 残段 = stop-trim 的 partial 末轮）——再直接
铺 block（`segment_len` = 该 block 前导连续 accept 数）、重 forward 取 student/teacher logp；**删掉** fork 大集
方案的反推 anchor / `min(block_size-1)` 截断 / 补块 / 去重（`dflash_student.py:484,519-611`）——因单流已无损给出
block 划分，无需反推。重建逻辑可直接复用 `scripts/opd/dspark_stream_unit.py:reconstruct`（T1 单测里已验证，含
stop-trim 场景）。落到本 repo `transformer_impl.py:_prepare_composed_*`。

**T5 单独测试方案**：
- **CPU 端到端（无需 GPU）**：喂已知 accept_state 流 → 重建 anchors + block plan → 断言与手算一致；对照本
  repo eval 侧 `draft_ops.py` 铺的 block 语义（slot i→anchor+1+i）一致；特别构造"跨多轮整块接受"样本，断言
  中间 round 的 anchor 不丢（fork 旧实现会丢）。
- **小 GPU 验证重 forward**：给一条真实 response + accept_state 流，重 forward draft/teacher，断言 logp
  shape 对、数值有限（非 NaN/Inf）、accept 段与 reject 段的 logp 都取到。

### 5.9 任务 T6 `[V]` — draft 权重同步接线（每 step 推 draft 回 sglang）

**做法**（改 verl，对齐 §5.2 与 Draft-OPD 参考）：
- **HYBRID 共卡 → 走进程内 IPC（附带好处）**：§5.0 定的 HYBRID 下 sglang 与训练**同进程**，推 draft 权重走
  `update_weights_from_tensor` 的 **CUDA IPC**（`weight_sync/utils.py`），比独立 server 跨进程更直接（无需 HTTP
  meta 中转）。
- 权重导出只取 draft trainable：`get_per_tensor_param(trainable_only=True)`；剥 `draft_model.` 前缀（仿
  `_map_opd_draft_weight_name`，`sglang_rollout.py:215`），非 draft 权重丢弃。
- 透传"只更 draft"标志：verl 现有 `sgl_update_weights`（`weight_sync/utils.py`）**不透传** draft 选择字段，
  需扩展；对接 T2 在 upstream 新增的 `draft_model_only` 入口。
- 键名与 upstream `Qwen3DSparkModel.load_weights` 对齐（与 T2 的键名对齐互为一体，联合验）。
- **时序**：在融合 `train_step` 的 update 之后、下一轮 rollout 的 wake 之前推权重，保证下一轮 rollout 用新 draft。

**T6 单独测试方案**（需 GPU + server，可与 T2 的 probe 复用）：
- 起 DSPARK server + verl 侧构造**一次** `update_weights` 调用（用训练侧真实 draft state_dict）：断言
  **（a）draft worker 权重被刷新**（`get_weights_by_name` 前后对比 / 或 sync 后 draft checksum == 训练侧）；
  **（b）target 不变**；**（c）draft 权重键 100% 命中**（`draft_tensor_count>0`，无 missing/unexpected）。
- 量级校验：仿 Draft-OPD `sglang_rollout.py:390` 的"若只更 draft 但 updated_tensor_count==0 直接报错"护栏。

### 5.10 集成测试方案（T1–T6 全部就绪后）

**目标**：完整 rollout→teacher scoring→update→weight-sync 闭环跑通若干 step，验证四类不变式。

1. **rollout 契约**：sglang 产出的 response 被现有 DSpark-OPD 管线正常消费（形状/mask/tensor-contract 与
   `docs/opd/tensor-contract.md` 一致），几个 step 不崩。
2. **单流自洽（信息完备性）**：`len(dspark_accept_state)` == `completion_tokens`（含 seed，精确）；
   `#(==COMMIT_BOUNDARY)` == `spec_verify_ct`（或 `== vct−1` 且流尾为非 boundary 残段 = stop-trim）；恰一个
   `PREFILL_SEED` 在 index 0；训练侧重建的 accept length == meta_info `spec_accept_length`；从 seed+boundary
   重建的 anchors + block plan 与 `draft_ops` 语义一致。
3. **权重同步生效（闭环关键）**：每 step 同步后**下一轮 rollout 用的是新 draft**——可观测证据二选一：
   （a）sync 后从 server `get_weights_by_name` 取 draft 权重 == 训练侧当前 draft；（b）随训练进行 server 侧
   accept length 单调变化（跟随 draft 变好/变化），而非恒定于初值。**target 全程不变。**
4. **训练健康度**：loss 有限、grad 正常、指标（accept length / KL / 反向 KL loss）与 **cache-response 基线
   同量级、可收敛**（对照 Stage-M3 的收敛口径）。
5. **HYBRID 显存/时序稳定（共卡关键）**：多 step 下 rollout 态峰值稳定 ≤ 80GB（不随 step 累积泄漏）；每 step
   wake→generate→sleep→update→推权重的时序不死锁、sleep 后训练态显存正常回落。

**建议脚本**：`scripts/opd/s_m2_integration_smoke.py`（或复用 `s4_smoke.py` 骨架）——跑 N=3~5 step 融合
`train_step`，落盘每 step 的：rollout 长度分布、accept_state 单流自洽校验结果、weight-sync 的 `draft_tensor_count`、
loss/grad_norm、server 侧 accept length 轨迹、**rollout/训练两态峰值显存**。全绿即 Stage-M2 验收通过，进入 Stage-M3。

### 5.11 附录：这个最小设计绕过的坑（对比"回传一大堆"方案）

之前调研发现，若照 fork 回传"被拒 token / q / teacher logp"，upstream 有三个硬坑：① `return_logprob` 对
DSPARK 抛 `ValueError`（`dspark_worker_v2.py:362`）；② q 只在采样路径算、且是局部变量（`dspark_verify.py:675`）；
③ folded/cuda-graph 路径 `corrected_logits=None`（`dspark_draft.py:283`），reject 内核还丢弃 `accept_index/
predicts`（`dspark_accept.py:159-172`）。**本设计只取 anchor + accept/reject 状态、logp 训练侧重算，这三个
坑全部不触发** —— 这是选它的根本理由。（upstream `dspark_observability.py` REQS dump 已能导出 anchor/
accept_len 到 `get_internal_state`，可作为"捕获逻辑"现成参考，但目的地是本地文件、非 `/generate`。）

---

## 6. Stage-M3 — DSpark-OPD 训练闭环 + 信息导出（⏳ 待做）

### 6.1 具体做法

用 upstream DSPARK rollout 跑通完整 DSpark-OPD 训练（复用现有反向 KL + `losses.py`/`compose_dspark_loss`），
指标与 cache-response 基线同量级、可收敛；draft 信息导出固化为 §5.1 的最小契约（anchor + accept/reject 状态），
accept/reject 两条流的 student/teacher logp 均由训练侧重 forward 恢复。

### 6.2 smoke-test 与用户手动测试

- smoke：小规模训练几个 step——不崩、loss 有限、grad 正常、指标与 cache-response 基线同量级。
- 用户手动：完整跑通一轮 DSpark-OPD 训练，观察收敛曲线与 cache-response 基线对齐；抽查导出的 draft 信息
  （anchor + accept/reject 状态）自洽。

---

# 第四部分 · 资源与操作速记

## 7. 资源与操作速记

### 7.1 ckpt / 机器

| ckpt | 归属 | 用途 |
|---|---|---|
| `Qwen/Qwen3-4B` | target（HF 已缓存）| 所有实验的 target |
| `/mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest` | **markov_rank=256 DSpark（可用）** | Stage-M1 全部验证 |
| `deepseek-ai/dflash_qwen3_4b_block7` | markov_rank=0（已弃用）| 无法驱动 upstream 原生 rejection |

- 机器：8×H100 80GB。CUDA toolkit `/opt/pytorch/cuda`（cu13, driver 13.2）。
- server env：`~/.venv/dspark-opd-sglang`；本地 eval 也用同一 env（transformers 5.12.1 能加载 repo DSpark modeling）。
- 起 server / 跑脚本前置：`export CUDA_HOME=/opt/pytorch/cuda; PATH/LD_LIBRARY_PATH` 前缀（启动脚本已自包含）。

### 7.2 本工作的脚本资产（`scripts/opd/`）

| 脚本 | 作用 | 用于 |
|---|---|---|
| `setup_env_sglang_upstream.sh` | upstream env 安装 | Stage-M0（§3）|
| `dspark_upstream_weight_keymap_check.py` | 离线权重键映射校验（47/47 CLEAN）| Stage-M1 前置（§4.0）|
| `dspark_rejection_equivalence_check.py` | 内核级 rejection 等价 + 无损（正确性金证据）| Stage-M1 验证点1（§4.1）|
| `dspark_local_eval_accept.py` | 本地 eval.py golden 基准（薄 wrapper，禁 confidence/entropy recorder）| Stage-M1 验证点2（§4.2）|
| `dspark_server_accept_probe.py` | server 侧 accept length（对齐 eval 口径拿 spec_verify_ct）| Stage-M1 验证点2（§4.2）|
| `spec_bench.py` | HTTP 吞吐/延迟/accept benchmark + baseline-vs-spec 加速比对照 | Stage-M1 验证点3（§4.3）|
| `launch_sglang_upstream_dspark.sh` | upstream DSPARK / baseline server 启动（默认 flashinfer）| Stage-M1/M2（§4.5）|
| `check_env.py` | 主 env（dspark-opd）训练侧运行时自检 | S0 smoke |
| `dspark_stream_unit.py` ✅ | T1 accept_state 纯函数 + anchor 重建 + 自洽 CPU 单测 | Stage-M2 T1（§5.4）|
| `dspark_draft_update_probe.py` ✅ | T2 draft-only 权重更新验证（HttpServerEngineAdapter + `/weights_checker` checksum：draft 变、target 不变、round-trip；`--draft-b` 两-ckpt 行为侧模式，q 可视化需按脚本头注释临时加回探针）| Stage-M2 T2/T6（§5.5/§5.9）|
| `dspark_draft_first_block_probe.py` ✅ | T2 行为侧：两 ckpt 对比首 block draft 预测概率 q（sym-KL/argmax 分歧，本地前向）| Stage-M2 T2（§5.5）|
| `s_m2_integration_smoke.py`（待建）| Stage-M2 集成 smoke（rollout→teacher→update→weight-sync 闭环）| Stage-M2 集成（§5.10）|

### 7.3 关键代码坐标速查

| 事实 / 资产 | 位置 |
|---|---|
| 本 repo 完整-draft-分布 rejection（正确性基准）| `deepspec/eval/base_evaluator.py:252-285` |
| accept length 汇总口径 | `base_evaluator.py:482`（`acceptance_length_sum/proposal_count`）|
| evaluate 采样（纯 softmax multinomial）| `deepspec/utils/sampling.py:logits_to_probs` |
| DSpark block 语义（slot i→anchor+1+i）| `deepspec/modeling/dspark/qwen3/modeling.py:452` |
| markov head（vanilla/gated/rnn）| `deepspec/modeling/dspark/markov_head.py` |
| upstream classic rejection（coin*q<p）| `third_party/sglang/python/sglang/srt/.../speculative/reject_sampling.py:71` |
| upstream DSpark verify 走 classic | `third_party/sglang/python/sglang/srt/.../dspark_components/dspark_verify.py:653-716` |
| upstream Qwen3DSparkModel + EntryClass | `third_party/sglang/python/sglang/srt/.../models/dspark.py:505,509` |
| upstream load_weights 前缀分流 | `third_party/sglang/python/sglang/srt/.../models/dspark.py:38-67` |
| upstream markov_rank>0 硬要求 | `third_party/sglang/python/sglang/srt/.../models/dspark.py:264-271` |
| upstream block/verify 解耦 | `third_party/sglang/python/sglang/srt/.../spec_info.py:222-224` |
| upstream draft 信息落点（Stage-M2）| `third_party/sglang/python/sglang/srt/.../managers/batch_result_processor.py:536-606` |
| fork target_only（只用 p，丢 q）| `third_party/sglang-dflash/sgl-kernel/csrc/speculative/speculative_sampling.cuh:77-92` |
| verl 标准 rollout 抽象 / 注册表 | `verl/workers/rollout/base.py:28,80,88`（`BaseRollout`/`_ROLLOUT_REGISTRY`/`get_rollout_class`）|
| verl 标准 sglang（sync 进程内 Engine）| `verl/workers/rollout/sglang_rollout/sglang_rollout.py:134,257,484` |
| verl 标准 sglang（async server + replica）| `sglang_rollout.py:1546`（`ServerAdapter`）、`async_sglang_server.py:51,208`、`replica.py:231-239` |
| verl HYBRID 共卡（rollout+训练 fused，Stage-M2 用）| `replica.py:146`（`init_hybrid`）；sleep/wake `async_sglang_server.py:437-451`、`sglang_rollout.py:287-295` |
| verl 异步 off-policy（已评估放弃）| `recipe/fully_async_policy/`（vllm-only、非 on-policy）、`recipe/one_step_off_policy/` |
| OPD on-policy 退化（ratio≡1，异步会破坏）| `recipe/dspark_opd/worker.py:332`（`old_log_prob=log_prob.detach()`）、`loss-design.md:63-68` |
| draft ~1.5B / NO_SHARD 决策（共卡放得下）| `loss-design.md §2.6.5`、memory `dspark-opd-multigpu-grad-sync-bug` |
| verl 标准 rollout 数据契约 | `sglang_rollout.py:792-806`（序列级 DataProto）|
| verl 标准权重同步（CUDA IPC + bucket）| `verl/workers/rollout/base.py:51`、`fsdp_workers.py:674,706,723`、`weight_sync/utils.py:40-103` |
| verl 权重同步 HTTP endpoint（async）| `http_server_engine.py:350-390`（sync）/`747-777`（async，只传 IPC meta）|
| verl 权重同步触发点（每 step）| `ray_trainer.py:2058` |
| upstream draft/target 权重路由（缺 draft-only）| `weight_updater.py:151-164`、`io_struct.py:1605`（仅 `disable_draft_model`）|
| 🔴 DSPARK 无 draft 权重更新入口（T2 要补）| `dspark_worker_v2.py:270-273`（`__getattr__` 回落 target）；draft runner 已就绪 `dspark_worker_v2.py:93-106`、`model_runner.py:378` |
| Draft-OPD 参考：draft-only 权重同步（fork）| verl `sglang_rollout.py:215,238,312`；fork sglang `io_struct.py:1356-1358`、`eagle_worker.py:1018` |
| dspark 当前融合 rollout（Stage-M2 要改，现绕过 sglang）| `recipe/dspark_opd/worker.py:344`（`train_step`）、`block_rollout.py:32,92`、`rollout.py:25,43`（registry 占位 / update_weights no-op）|
| Draft-OPD 参考：标准多 RPC fit（未重写）| `Draft-OPD/verl/.../main_ppo.py:299`、`ray_trainer.py:1691,1792,1899,2021,2057` |
| Draft-OPD 参考：teacher = 独立 replica（非 co-resident）| `Draft-OPD/verl/.../main_ppo.py:203-210`、`ray_trainer.py:1017-1025`、`teacher_model.py:75-89`、`async_sglang_server.py:788` |
| dspark co-resident teacher（M2 保留不动）| `recipe/dspark_opd/worker.py:152`（`_build_teacher`）|
| Draft-OPD 参考：起 server + 抽 meta_info | `Draft-OPD/verl/.../async_sglang_server.py:208,589-627`、`sglang_rollout/utils.py:29-53` |
| Draft-OPD 参考：大集字段生产（fork sglang）| `Draft-OPD/sglang-dflash/.../managers/customized_info_utils.py:88-171` |
| anchor 反推（Stage-M2 最小集可删）| `Draft-OPD/verl/.../models/transformers/dflash_student.py:484,519-611` |
| upstream sglang（submodule）| `third_party/sglang` → fork `happyfanta00/sglang` 分支 `dspark-opd-patches`（base HEAD 7e7129a，T1/T2 改动 commit `5a309fe`）|

---

## 附：历史路线（已放弃，保留备查）

早期方案是**在 fork `third_party/sglang-dflash` 上手写** DSPARK 支持（新建 `models/dspark.py` +
`dspark_worker.py` 等、约 5 新建 9 编辑文件），并做了 markov-off 链路（Stage-1a/1c）：backbone 与 DFlash
同构、greedy 逐 token 无损、accept≈2.0（DFlash 的 86–91%）。但因 fork 的 `target_only` verify 不满足
完整-draft-分布 rejection 正确性标准（§2.3），**该路线已整体放弃**，改用 upstream 原生 DSPARK。fork 侧
代码（vendored `third_party/sglang-dflash` + 手写的 `dspark.py`/`dspark_worker.py`）已从本分支 `git rm`
删除——如需查阅历史快照见 git tag `backup/pre-fork-cleanup`。

fork 方案踩过的坑（仍有参考价值）：DSpark slot i 预测 `anchor+1+i` 比 DFlash 前移一格，直接复用
DFlashWorker 取 `[:,1:]` 会错位导致 accept≈0.02，需取 `[:, :block_size-1]` 对齐；fork 固定块长下无法用满
block_size 个提议（cuda graph verify buffer 尺寸冲突）——这些在 upstream 原生 block/verify 解耦下都不再是
问题。
