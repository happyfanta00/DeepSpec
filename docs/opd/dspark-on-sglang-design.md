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
| **Stage-M0** | 统一 env 安装（sglang HEAD + verl 训练栈同 env，§3）| ✅ 完成（2026-07-17 补 verl 栈：`get_rollout_replica_class("sglang")` 可解析）|
| **Stage-M1** | 正确性 + 接受率 + 加速比三项验证（§4）| ✅ 全部达成 |
| **Stage-M2** | verl rollout 接 upstream DSPARK sglang（直驱 `HttpServerEngineAdapter`，tp=8 独立 server actor 共驻 8 卡 + 小池常驻不错峰）+ draft 信息最小契约导出 + draft 权重回灌；拆 7 子任务（T3→T3a/T3b）+ 集成（§5）| ⏳ 进行中（T1+T2✅ server 待测；**T3a✅ 实测通过**（tp=8+投机解码+常驻共存 51GB<80）；**T3b✅**（accept_len 5.15 perfectblend 合理，做法X纯prompt源）；**T4✅**（accept_state→response-indexed张量）；**T5✅**（去Phase-1+重建block plan+full-vocab，CPU端到端过）；**T6✅ 已实现**（rank0 client 回灌 draft_model_only，tp_size份 serialize，T2已验通路）；**T1–T6 全落码，待集成 GPU smoke §5.10**）|
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

- **env = 统一 env**：`~/.venv/dspark-opd-sglang` **同时装 sglang HEAD + verl 训练栈**（M2 T3 起 verl
  HYBRID rollout 在本 env 内把 sglang 作为同 env 的 Ray actor+子进程拉起，故二者必须同 env 可 import；
  见 §5.0/§5.6）。**主 env `~/.venv/dspark-opd` 永久不动**（纯训练回退基线，transformers 5.10.2 /
  torch 2.9.1+cu128）。
- **本机满足 cu13**：toolkit `/opt/pytorch/cuda` 是 13.0，driver 13.2；`CUDA_HOME=/opt/pytorch/cuda`。
- **走最小子集路线**（flashinfer 全量非必需；DSPARK spec 链只需 torch/triton/msgspec）。两个坑：
  1. rust 扩展报错 → `export SGLANG_BUILD_RUST_EXTS=none`（`python/setup.py:46`）跳过，DSPARK 不需要。
  2. 逐步补 import 依赖（pybase64/IPython/gguf/openai-harmony 等）。
- **安装脚本**：`scripts/opd/setup_env_sglang_upstream.sh`（STEP 1 torch → STEP 2 sgl-kernel →
  STEP 3B flashinfer / 3A 最小子集兜底 → **STEP 3C verl core 运行时依赖**（ray/tensordict<=0.10/
  torchdata/codetiming/hydra-core/accelerate/peft/…；不碰 torch/tf/numpy）→ STEP 4 editable 装 sglang →
  **STEP 4B editable 装 verl（`--no-deps`）** → STEP 5 统一 env 冒烟 → STEP 6 冻结）。deepspec 靠
  `PYTHONPATH` 不 pip 装。**numpy 保留 2.x**（verl 声明 numpy<2 但 sglang HEAD 需 ≥2；`--no-deps` 装 verl
  故 numpy 不动，verl 0.7 多数功能在 numpy 2.x 可跑）。
- **verl↔sglang 版本 skew 修复**：verl 0.7 按 sglang **0.5.2** 写，跑 HEAD 时几处 import 符号移位；在
  recipe `__init__.py` 加 compat shim（与既有 transformers `AutoModelForVision2Seq` 别名同处）——注入
  `http_server._launch_subprocesses`（3-tuple wrapper）+ re-export `sglang.srt.utils.{get_open_port,
  get_local_ip_auto}`（HEAD 移进 `.network` 子模块）。**顺序坑**：`import sglang` 会替换
  `sys.modules["transformers"]`、冲掉别名，故 `__init__` 里先 sglang shim、再 transformers 别名。纯训练
  env（无 sglang）该 shim 静默 no-op、零回归。
- fork 的旧安装脚本 `setup_env_sglang.sh`、`pip-freeze-sglang.txt` 等已随 fork 路线一并删除。

### 3.2 smoke-test 方法与结果

**smoke-test = 安装脚本内置 STEP 5**：先 `import recipe.dspark_opd`（跑 compat shim）→ 再 import
原生 DSPARK 模块 + verl 训练栈 + deepspec，且 `get_rollout_replica_class("sglang")` 可解析，全成功打印
`[Stage-M0] UNIFIED ENV IMPORT OK`。

```python
# (1) sglang 原生 DSPARK：sglang / reject_sampling / dspark_components.* / models.dspark
# (2) verl 训练栈：ray / verl / verl.workers.rollout.replica / ...sglang_rollout.async_sglang_server
# (3) deepspec.modeling.dspark.qwen3（PYTHONPATH 提供）
# (4) get_rollout_replica_class("sglang") -> SGLangReplica（统一 env 健康探针；T3 实际直驱
#     HttpServerEngineAdapter、不走 replica，但此项确认 verl sglang 栈完整可 import）
```

**结果（✅ 2026-07-17）**：统一 env 全绿——sglang 原生 DSPARK 模块 + verl（含 `SGLangReplica`/
`async_sglang_server`）+ deepspec draft modeling 同 env 共存；shim 实际生效项
`['_launch_subprocesses','utils.get_open_port','utils.get_local_ip_auto']`。纯训练 env 复验 shim no-op、
`DSparkRollout` 路径零回归。

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

### 4.5b DRAFT backend 改用 fa3 修 flashinfer 长跑 hang（Stage-M3，2026-07-21 实测）

**现象**：`draft=flashinfer` 下长训（1 epoch）跑到 step ~88 / ~110 必 hang——一张卡 GPU util 100%、其余 7 张卡在 TP broadcast 等它，300 秒后 sglang watchdog `SIGQUIT` 杀 server，训练 worker 下次 `generate` 收 `ConnectionError: RemoteDisconnected` 崩。

**根因**（py-spy + 读码定位）：卡点是 flashinfer **每次 cuda graph replay 前的 metadata-prep**（`init_forward_metadata_out_graph → plan()`）里的 D2H 同步（`plan()` 的 `.to("cpu")`/`.item()`）。`_out_graph` 不是"batch 超出 graph 覆盖"，而是"graph 命中后 replay 前那次图外 eager plan"（flashinfer plan 无法录进 cuda graph）。长-decode shape 上某个 GPU attention kernel 跑飞不返回，`.cpu()` 同步就永久阻塞 → 单 rank 卡死 → 全 TP 死锁。decode graph 只固化 bs（≤512）不固化 seq_len，所以长序列不掉出图、而是在命中图的正常步里挂。

**修法**：DRAFT attention backend 由 flashinfer 改 **fa3**（`speculative_draft_attention_backend="fa3"`，target 仍 flashinfer 做 verify）。fa3 的 target_verify graph replay 在 **eagle_topk≤1**（DSPARK 恒满足，默认 1）下是 **sync-free** 的（`needs_cpu_seq_lens=False`，`flashattention_backend.py:173,2548-2583` 全 device 操作、无 `.cpu()/.item()`）——**保留 spec-decode cuda graph（无 eager 3x 慢），且消除会挂的同步点**。实测长训不再 hang。

- 落地：`sglang_server.py` `_DEFAULT_DRAFT_ATTN_BACKEND="fa3"`（env `DSPARK_SGLANG_DRAFT_ATTN_BACKEND` 可 A/B 覆盖回 flashinfer 复现 / triton 对比）。
- 为何不用其它：`disable_decode_cuda_graph`（走 eager）能规避但 rollout 慢 3x；triton 保留 graph 但 spec-decode 长序列有历史 crash 隐患（§4.5，DSPARK `custom_mask=None` 或可规避但未证）；给 flashinfer target_verify 补 split-kv 需改核心且 flashinfer split-kv+graph 官方不保证。fa3 是"保留 graph + 风险最低"的解。
- 前提：不可设 `--speculative-eagle-topk>1`（否则 fa3 走含 `.item()` 的 topk>1 分支，同步卡点回来）。

---

## 5. Stage-M2 — verl rollout 接 upstream DSPARK sglang（⏳ 待做，已出方案）

**目标**：把 DSpark-OPD 训练里"现场由 draft nn.Module block 采样出的 response"替换为"upstream DSPARK
sglang 现场投机解码产生的 response"，其余训练逻辑不变、tensor-contract 不变（`docs/opd/tensor-contract.md`）；
同时用**最小契约**导出 draft 信息。

### 5.0 接入拓扑与起点（直驱 sglang HttpServerEngineAdapter）

> **本节先厘清"标准做法"与"当前起点"，是 §5.1 起全部子任务的前提。已核对 verl 本体（3 个 verl 树）、
> Draft-OPD/Rethink-OPD 参考、当前 dspark_opd recipe 多处源码。**

**verl 的标准 sglang rollout 抽象**（`verl/workers/rollout/base.py:28,80,88`）：`BaseRollout` 基类 +
`_ROLLOUT_REGISTRY` + `get_rollout_class(name, mode)`。sglang 有两条官方路径：
- **sync（进程内 Engine，默认/主流）**：`SGLangRollout` 在 worker 进程内嵌 `sglang.Engine` 子类，
  token-in-token-out，不走 HTTP（`sglang_rollout.py:134,257,484`）。**但其 `_init_inference_engine` 用硬编码
  whitelist、不 splat `engine_kwargs`，spec-decode 参数被静默丢弃——排除。**
- **async（server 模式）**：`ServerAdapter(BaseRollout)` + `SGLangReplica(RolloutReplica)` + Ray actor
  `SGLangHttpServer`（`async_sglang_server.py:51`、`replica.py:231-239`），控制面走 HTTP、生成走进程内
  `tokenizer_manager.generate_request`。**能传 spec（`**engine_kwargs`），但 `init_hybrid` 强制 async
  worker + `ServerAdapter` + 全量灌权，与融合设计冲突——不直接复用其编排壳。**
- ⚠️ **verl 本体对投机解码无一等公民支持**（vendored 0.7 全树 grep `speculative/eagle/draft_model` 零命中；
  spec example 只在 Draft-OPD 0.8，且只走 async-server 路），只能靠 `engine_kwargs.sglang` 透传底层 sglang
  的 spec-decode 参数。**我们取其"底层引擎类 `HttpServerEngineAdapter` + spec 透传"，去掉其"replica/
  AgentLoop 编排壳"——见下方决策。**

**决策（用户已定，2026-07-17 定稿，修订原"复用 SGLangReplica/init_hybrid"方案）：在融合 `train_step`
内直接驱动 sglang `HttpServerEngineAdapter`（tp=8 单 server 子进程共驻 8 卡），绕开 verl 的 replica/
AgentLoop 编排壳。** 依据见本节末"机制选型调查"。要点：
- **句柄**：`HttpServerEngineAdapter`（`verl/workers/rollout/sglang_rollout/http_server_engine.py:194`）
  ——它是 verl **官方 async-server 路径的底层同一个引擎类**，`__init__` 里 `ServerArgs(**kwargs)` +
  `launch_server_process` 起一个 sglang server 子进程。我们直接构造它、自管生命周期，**不套**
  `SGLangReplica`/`AgentLoopManager`/`ServerAdapter`（那套要求 async worker、拆融合循环、每 step 全量灌权，
  与 M2 融合设计正面冲突）。
- **spec 参数透传**：`HttpServerEngineAdapter(model_path=target, speculative_algorithm="DSPARK",
  speculative_draft_model_path=draft, tp_size=8, ...)` → `ServerArgs(**kwargs)` 原样收（不像 verl 进程内
  sync 路 `_init_inference_engine` 用 whitelist 把 spec 参数丢弃，`sglang_rollout.py:440-472`）。
- **生成**：`engine.generate(input_ids=prompt, sampling_params=...)` → 返回 raw server JSON dict（含
  `output_ids` + `meta_info`，后者带 T1 的 `dspark_accept_state` 流）。**这与 T2 probe
  （`dspark_draft_update_probe.py:312`）用的是同一个类、同一条路，已实测 generate + 权重回灌可用。**
- **draft 信息**：复用 sglang 的 `customized_info → meta_info` 通道（§5.1）；T1 生产端已在 submodule
  sglang 落地。T4 从 `generate` 返回 dict 的 `meta_info["dspark_accept_state"]` 抽流（不再需要 verl
  `ServerAdapter.generate` 的 `extra_fields` 搬运，因我们直接拿 dict）。
- **权重回灌（T6）**：`engine.update_weights_from_tensor(draft_named_tensors, draft_model_only=True)`
  （CUDA-IPC，T2 已验证 draft 变/target 冻结/round-trip）。

**机制选型调查（3 个 verl 树 exhaustive，2026-07-17）——为何不走 init_hybrid / 进程内 sync**：
- **spec-decode 在 vendored verl 0.7 里完全没有官方 example**（examples/recipe/tests/docs/config 全零命中
  `speculative_*`/eagle/mtp）。唯一有 spec-decode example 的是 **Draft-OPD 的 verl 0.8**，且**只走
  async-server 路径**（`AsyncHttpServerAdapter`+`_launch_subprocesses`，`engine_kwargs.sglang.
  speculative_*` 经 `async_sglang_server.py:279` 的 `**engine_kwargs` splat 进 `ServerArgs`）——**从不走进程内 sync 路。**
- **verl 进程内 sync（`SGLangRollout`）传不了 spec 参数**：`_init_inference_engine`（`sglang_rollout.py:
  440-484`）用硬编码 whitelist、不 splat `engine_kwargs`，`speculative_*` 被静默丢弃（0.7 未 patch）。⇒ 排除。
- **verl async `init_hybrid`（Draft-OPD 走这条）**：唯一 verl-native 能传 spec 的路，但强制
  `AsyncActorRolloutRefWorker`（我们是 sync 基类子类，无 `wake_up/sleep`）+ `self.rollout` 须
  `ServerAdapter` + `rollout_mode()` 每 wake 全量灌权（与 T6 draft-only 冲突）+ generation 移出 worker
  （破坏融合"中间产物留显存只回标量"）。⇒ 与 M2 融合设计冲突大，排除。
- ⇒ **选 `HttpServerEngineAdapter` 直驱**：它就是官方 async-server 路径的底层引擎类，spec 透传同源、
  权重回灌 T2 已验；唯一"自己扛"的是 server 生命周期编排（官方由 `AgentLoopManager`/replica 代管，我们在
  worker 里显式管，见 §5.6 生命周期）。

**⚠️ 当前起点（务必先知）**：当前 dspark_opd recipe **默认并不走 sglang，也不经 verl 的
`generate_sequences` 入口**。默认路径是融合单 RPC `DSparkActorRolloutRefWorker.train_step`
（`worker.py:344`），在一个 RPC 内做 rollout→teacher scoring→update，rollout 段直接对 attach 的 draft
nn.Module 做 block-parallel 前向采样（`block_rollout.py:32,92`）；`DSparkRollout(BaseRollout)`
（`rollout.py:25`）目前只是 registry 合规占位（仅旧 3-RPC 路径用到）。cache 只提供输入特征
（`target_hidden_states`），非读"现成 response"。teacher（target）是 **co-resident** 建在 actor worker
进程内（`worker.py:152` `_build_teacher`），非独立 worker。

**架构取舍（用户已定，勿再动训练骨架）**：Stage-M2 的实质工作是把 rollout 段从"draft module 现场 block
采样"切到"upstream DSPARK sglang 现场投机解码"。**只对齐 rollout 侧的 sglang 接入这一件事——在融合
`train_step` 内直接持有 `HttpServerEngineAdapter` 句柄、调 `generate` 当 response 来源，保留融合训练循环 +
co-resident teacher 不变。**
- 依据：已核实 Draft-OPD 参考走 verl **标准多 RPC `RayPPOTrainer.fit()`（未重写）** + teacher 是**独立
  rollout replica（独立 GPU 池 + 独立 sglang server，`is_teacher_model=True`）** + async `init_hybrid`。**完全
  对齐它 = 连带把融合 `train_step` 拆回标准多 RPC + co-resident teacher 拆成独立 replica + worker 改 async——
  多处大重构，远超"换 response 来源"的目标，风险高。** 故 M2 不走这条。
- **只复用引擎类、不复用编排壳**：verl 的 `init_hybrid`/`RolloutReplica`/`ServerAdapter`/`AgentLoopManager`
  是绑定 async worker 的编排壳，**不复用**；只复用其底层引擎类 `HttpServerEngineAdapter`（官方 async-server 就
  是用它起 server），在融合 `train_step` 内当"起 sglang + 发 generate + 回灌权重"的句柄直接构造。T3（§5.6）
  即按此做，生命周期由 worker 自管（§5.6）。

**GPU 资源布局（用户已定）：sglang server tp=8 单进程共驻全 8 卡，NO_SHARD 保留、不 offload，
rollout↔training 靠 memory_occupation release/resume 错峰。**

- **模式**：**一个** sglang server 子进程（`HttpServerEngineAdapter(tp_size=8)`）张量并行铺满 8 卡，与 actor
  FSDP 训练 **colocate 在同一批 8 卡上时分复用**。server 子进程内部自建一套 8 卡 NCCL 组做 TP，与训练的
  `torch.distributed`（NO_SHARD DP）**隔离两套**（不共用通信组，规避 external_launcher × NO_SHARD 潜在冲突）。
  **不**走"rollout 独占卡"（阶段性闲置且不必要），**不**走异步 off-policy（§已评估放弃：OPD 严格 on-policy，
  `old_log_prob=log_prob.detach()` 使 `ratio≡1`，异步 staleness 破坏该近似，属改算法而非改部署）。
- **为何 tp=8 单 server 而非 tp=1×8**：tp=8 把 target 4B + draft 1.5B 权重**分片到 8 卡**（每卡仅 1/8），
  rollout 态每卡引擎显存骤降，与训练态错峰更宽松；且只有一个 server、一套端口、一处生命周期，编排最简。
  （tp=1×8 = 每卡一份完整 target+draft，显存最紧，本项目不取。）
- **显存账（H100 80GB / bf16，粗算，⏳ 待 §5.6 实测坐实）**：draft 仅 **~1.5B**（embed/lm_head 冻结），
  teacher 4B 冻结（无梯度/优化器）。
  - **训练态**（sglang release、KV+权重显存吐出）：draft NO_SHARD footprint（权重~3G + 梯度~3G + Adam fp32
    三件套~18G）~24GB + co-resident teacher 8GB ≈ **~32GB/卡**——与今天单卡训练态同，已验证能跑。
  - **rollout 态**（sglang resume）：tp=8 下引擎权重分片（target+draft 合计 ~5.5B/8 ≈ 每卡~1.4GB bf16）+
    KV 池 + cuda graph，`mem_fraction_static` 视余量调（tp=8 权重分片后可给 KV 更多）。
  - **共卡合计**：训练常驻 ~32GB + rollout 态引擎（分片权重 + KV），**目标 ≤ 80GB** ✓ → 期望两态同时驻留、
    无需 offload、NO_SHARD 不用改。**这份粗算是 §5.6 "第一产出=显存实测 gate" 要坐实的前提。**
- **为什么 NO_SHARD 保留**：draft 只 1.5B，共卡放得下，故那个"为多卡梯度同步正确性刻意选 NO_SHARD"的决策
  （`loss-design.md §2.6.5`、memory `dspark-opd-multigpu-grad-sync-bug`）**不用动**，风险归零。
- **release/resume 错峰（= sleep/wake）**：rollout 段前 `resume_memory_occupation()` 收回 KV+权重显存、
  段后 `release_memory_occupation()` 吐出给训练激活。子进程**全程不死**，只吐/收显存（依赖 sglang
  `enable_memory_saver=True`，`torch_memory_saver` 机制），近零成本。⚠️ **release/resume 这条 T2 probe 未测过**
  （T2 只测 generate + 权重回灌），§5.6 显存 gate 要顺带验证其时序 + 显存回落。
- ⚠️ **安全阀（未来 OOM 再启用，现在不预防性优化）**：余量靠"draft 1.5B + tp=8 权重分片 + 序列短
  （resp=64/anchor=32，激活小）"。若未来长序列 / 大 batch 吃满余量，**再考虑切 FULL_SHARD 分片或 rollout 态
  param/optimizer offload**——用户已定此为延后项。

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

### 5.3 任务拆分总览（7 个子任务 + 集成测试）

Stage-M2 改动横跨 upstream sglang 与 verl 两侧，拆成可**独立开发、独立测试**的子任务，最后一次集成。
`[U]`=改 upstream sglang，`[V]`=改 verl。

> **T3 拆分（2026-07-17）**：原 T3 揉了"起 server + 8 卡可见性 + prompt 抽取 + generate + 新序列重建 +
> 生命周期"太多耦合，难独立测、难定位。按自然接缝拆成 **T3a（server infra，纯 infra 零训练依赖）** +
> **T3b（train_step 接线，依赖 T3a）**——把"最 novel、最 risky、且能零依赖独立验证"的 server actor 单独切出。

| 任务 | 侧 | 内容 | 依赖 | 详见 |
|---|---|---|---|---|
| **T1** | `[U]` | draft 信息导出（单条 per-token `dspark_accept_state` 流 → `customized_info` → meta_info；anchor 训练侧推）✅ | — | §5.4 |
| **T2** | `[U]` | DSPARK draft-only 权重更新入口（补 `update_weights_from_tensor` + `draft_model_only` 标志）✅ | — | §5.5 |
| **T3a** | `[V]` | **DSPARK server infra**：独立 Ray actor（NOSET + 拼 8 卡 `CUDA_VISIBLE_DEVICES` + NodeAffinity 共驻训练 8 卡、不额外占 Ray GPU）内起 tp=8 `HttpServerEngineAdapter(DSPARK)`；暴露 host:port / generate / release·resume / update_weights / shutdown；driver 建一次、fit 末 shutdown | — | §5.6a |
| **T3b** | `[V]` | **train_step rollout 接线**：worker 拿 server host:port 当 client → `train_step` Phase-1 换成 prompt 抽取→resume→generate→release→收 response+accept_state → 新序列重建（`recompute_target_hidden_states`）| T3a | §5.6b |
| **T4** | `[V]` | draft 信息传输（generate dict `meta_info["dspark_accept_state"]` → response-indexed 张量 `[B×n,R]` pad=-1 + `response_lengths`）✅ CPU 单测过 | T1, T3b | §5.7 |
| **T5** | `[V]` | draft 信息消费（**去 Phase-1** + accept_state 重建 block plan + **Phase-2/3 返 full-vocab** `[B,A,blk,V]`；reject 位记 q/p 不进 loss=选项B）✅ CPU 端到端过 | T4 | §5.8 |
| **T6** | `[V]` | draft 权重同步接线（每 step update 后 rank0 经 client `update_weights_from_tensor(draft_model_only)` 回灌，导 actor_module 剥冻结件、tp_size 份 serialize）✅ | T2, T3a | §5.9 |
| **集成** | 两侧 | 完整 rollout→teacher→update→weight-sync 闭环 | T1–T6 | §5.10 |

**依赖图**：
```
[U] T1 (信息导出) ✅ ───────────────────────────────────┐
[U] T2 (draft 权重入口) ✅ ──────────────┐               │
[V] T3a (server infra) ──┬──→ T3b (train_step 接线) ──┬──→ T4 (传输) ──→ T5 (消费) ──┐
                         │                             └─────────────────────────────┤
                         └──────────────────────────────→ T6 (权重同步)──────────────┼─→ 集成 §5.10
                          (T6 需 T2 + T3a)                                            ┘
```
- **T1/T2/T3a 无相互依赖，可三线并行起步**（T1、T2 是 sglang 源码改动✅，T3a 是 verl 侧独立 server actor）。
- T3b 需 T3a（有 server 可打）；T4 需 T1（有流）+ T3b（train_step 拿到 dict）；T5 接 T4；T6 需 T2（draft 入口）
  + T3a（有 actor 的 `update_weights_from_tensor` 可调，CUDA-IPC 跨进程同物理卡已确认可用）。
- **建议顺序：T3a（先坐实 server infra）→ T3b（用旧随机 anchor 占位即可测）→ 再 T4→T5 与 T6 并行 → 集成。**

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

### 5.6a 任务 T3a `[V]` — DSPARK server infra（独立 Ray actor 起 tp=8 server，共驻训练 8 卡）✅ 已实现并实测

> **✅ 实测通过（2026-07-17，`s_m2_t3a_server.py`，8×H100）**：tp=8 单 server 共驻 8 卡起得来、投机解码正常
> （aggregate accept_len 6.15，gsm8k 金标准 ~6.2 量级，accept rate 0.66~0.76）、4/4 accept_state 流结构自洽、
> 生命周期（actor 建/shutdown）正常。落地文件：`recipe/dspark_opd/sglang_server.py`
> （`DSparkSglangServer` actor + `build_dspark_sglang_server`）。
>
> **✅ 常驻方案显存实测坐实（`mem_fraction_static=0.15`）**：rollout 态峰值 **17.1GB/卡**（vs 0.6 时 53GB）；
> `--sim-train-gib 32`（模拟 FSDP draft+teacher）共存峰值 **51.3GB/卡 < 78GB cap**、余量 ~28GB，且共存时 engine
> 仍正常投机解码。⇒ **小池常驻方案成立，无需 release/resume 错峰**（§5.6b）。`max_running_requests=48`、
> `max_total_num_tokens=52万`，训练并发绰绰有余。
>
> **⚠️ 血泪坑：release/resume 权重必须开 `enable_*_weights_cpu_backup`（否则 resume 后权重全变随机）**：
> 首版漏设 → `release_memory_occupation` 释放 WEIGHTS 物理页、`resume` 重映射**全新垃圾页**（`torch_memory_saver`
> 只在 `enable_cpu_backup=True` 时 pause 拷 GPU→CPU、resume 拷回；sglang `load_model_utils.py:218`）。实测坐实：
> resume 后 **target 290/290 + draft 46/46 全变随机** → 乱码输出 + accept 8.00 满块。**修法**：`HttpServerEngineAdapter`
> 加 `enable_weights_cpu_backup=True` + `enable_draft_weights_cpu_backup=True`（DSPARK draft 在独立 runner，两个都要；
> 与 Draft-OPD composed-DFLASH-student server `async_sglang_server.py:315-329` 一致）。
>
> **release/resume 耗时观测（T3b 错峰决策依据）**：实测 release ~2s + resume ~3s ≈ **每轮 ~5s** GPU↔CPU 权重拷贝
> （target+draft ~5.5B bf16 ≈ 11GB）。若 train_step 本身才几秒，per-step 错峰开销显著 → T3b 需评估"常驻不错峰
> vs 每步 release/resume"（§5.6b）。

> **机制定稿（2026-07-17，修订原 init_hybrid 方案，依据见 §5.0 "机制选型调查"）**：直接持有
> `HttpServerEngineAdapter` 句柄驱动 sglang DSPARK 引擎，**不套** verl 的 `SGLangReplica`/`init_hybrid`/
> `ServerAdapter`/`AgentLoopManager` 编排壳。此句柄类 = 官方 async-server 路径的底层同一个引擎类，spec 透传同源、
> 权重回灌 T2 已验，**零 vendored-verl core 改动、零 sglang core 改动**。

> **⚠️ 关键约束（核实源码后修正原"rank0 在训练 worker 内起 tp=8 server"的错误设想）**：dspark 训练是
> **8 个 Ray actor、每个 `num_gpus=1`**，且 recipe **未设** `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`
> → Ray 给每个训练 worker **只暴露它自己那 1 张卡**（`base/worker.py:248-256`）。故**任何训练 worker 进程都
> 起不了 tp=8 server**（看不见 8 卡）。⇒ tp=8 server 必须放在一个**独立 Ray actor**里，靠
> `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` + 显式拼 8 卡 `CUDA_VISIBLE_DEVICES` + `NodeAffinity`
> 获得全 8 卡可见性——**这正是 verl 官方 async 起 server 的物理放置法**（`async_sglang_server.py:284-300`），
> 我们复刻其**放置**、但不套其 replica/AgentLoop 编排。

**做法**（新增 `recipe/dspark_opd/sglang_server.py` + driver 侧接线，纯 infra、不碰 DSpark tensor-contract）：

**① 独立 server actor**（仿 `async_sglang_server.py:284-300` 的 actor options）：
```python
@ray.remote  # 不声明 num_gpus —— 共享训练已占的 8 卡（靠 NOSET+显式 CUDA_VISIBLE_DEVICES 获可见性）
class DSparkSglangServer:
    def __init__(self, *, target_path, draft_path, cuda_visible_devices, port, mem_fraction_static):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices   # "0,1,...,7"
        from sglang.srt.entrypoints.http_server_engine import HttpServerEngineAdapter
        self._engine = HttpServerEngineAdapter(
            model_path=target_path, speculative_algorithm="DSPARK",
            speculative_draft_model_path=draft_path,                # gamma 自动从 ckpt config 读
            speculative_draft_attention_backend="flashinfer",
            attention_backend="flashinfer",                         # §4.5：triton spec 长序列会崩
            tp_size=8, mem_fraction_static=mem_fraction_static,
            enable_memory_saver=True,                               # release/resume 依赖
            trust_remote_code=True, port=port)
    def get_address(self):  return (host, port)                    # 供训练 worker 当 client
    def generate(self, input_ids, sampling_params):  return self._engine.generate(...)
    def release(self):  return self._engine.release_memory_occupation()
    def resume(self):   return self._engine.resume_memory_occupation()
    def update_weights(self, named, draft_model_only=True, flush_cache=True): ...
    def shutdown(self): self._engine.shutdown()
```
- actor 用 `.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
  runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}})` 落到训练节点、拿全卡可见性。
- **driver（`task_runner.py`）在 `trainer.init_workers()` 后建一次**：`__ray_call__` 收 8 个训练 worker 的
  `(node_id, CUDA_VISIBLE_DEVICES)`（仿 `async_sglang_server.py:261-278`）拼出 `cuda_visible_devices` + node，
  建 `DSparkSglangServer` actor，拿 `get_address()` 存下（T3b 用它当 client 地址）。

**② 生命周期**：driver 建一次（长驻）；`DSparkTrainer.fit()` 末尾（final save 后）`ray.get(server.shutdown.remote())`
+ `ray.kill(server)`。⚠️ verl Worker 基类**无 teardown 钩子**、trainer fit 也无 finally——核实确认，故必须**显式**
在 fit 末尾关，另加 `atexit` 兜底防僵尸子进程。

**T3a 单独测试方案（纯 infra，零 train_step / 零 tensor-contract 依赖）**——本质是"把 T2 probe 升级成常驻 actor"：
- `scripts/opd/s_m2_t3a_server.py`：driver 建 `DSparkSglangServer` actor（共驻当前 8 卡）→ 发 1~8 个 gsm8k
  prompt（**金标准口径** temp=1.0/chat template/enable_thinking=False，§2.4）→ 断言：拿回 `output_ids`
  形状/EOS 合理、`meta_info` 带 `dspark_accept_state`（`len==len(output_ids)`）、accept length 与 §4.2 golden
  同量级（~6.2）。**期望末行 `RESULT: T3a SERVER OK`。**
- **release/resume 验证**（T2 未覆盖的新点，依赖 `enable_memory_saver=True`）：`resume→generate→release` 一轮，
  用 `nvidia-smi` 采样断言 release 后该 server 的 KV/权重显存回落、resume 后可再 generate。
- 复用 §4.3 `spec_bench.py` 口径确认 tp=8 server 吞吐/accept 与 Stage-M1 一致（无回归）。
- **可复用 T2 资产**：`dspark_draft_update_probe.py:312` 已是"构造 `HttpServerEngineAdapter(DSPARK)` +
  generate + update_weights"的最小工作范例，T3a actor 内部直接借其 engine 构造。

### 5.6b 任务 T3b `[V]` — train_step rollout 接线（client + generate + 新序列重建）

**★ prompt 数据源（用户拍板 2026-07-17，做法 X）：换纯 prompt 语料 + apply_chat_template，不从 cache 反切。**
- **为何**：cache（`perfectblend_train_regen.jsonl` 建）存的是 **prompt+已生成 response** 的完整对话，且
  enable_thinking=False 的空思考壳 `<think>\n\n</think>\n\n` 落在 **response 段**（loss_mask=1）。从
  `input_ids[:first loss_mask==1]` 反切 prompt 会**丢掉这个壳** → rollout prompt 分布偏离金标准 → **accept_len
  掉到 3.76**（实测，vs golden ~6.2）。
- **做法 X**：新增 `DSparkPromptDataset`（`dataset.py`）从**原始纯 prompt 语料**
  `train_datasets/perfectblend_train.jsonl`（每条只有 user、无 assistant）读，`encode_chat_messages(
  add_generation_prompt=True, enable_thinking=False)` 现搭 prompt——**与 evaluate 金标准逐字节一致**（含
  `<think>\n\n</think>\n\n` 壳）。CPU 实测：3 条 prompt 全部以该壳结尾、loss_mask 全 0。
- **接线**：`task_runner` 在 `DSPARK_SGLANG_ROLLOUT=1` 时把 `data.custom_cls.name` 从 `DSparkCacheDataset`
  换成 `DSparkPromptDataset`（config 新增 `data.dspark.prompt_jsonl_path`）；`extract_prompts` 对 prompt 数据集
  用 `attention_mask` 长度取 prompt（loss_mask 全 0），对旧 cache 仍按 first-loss_mask 切（兼容）。**默认
  （flag 未设）走 cache 路，零回归。**

**三个落地决策（用户拍板 2026-07-17）**：
1. **n 条独立 rollout（采纳）**：每 prompt 生成 `rollout.n` 条**不同** response（temp=1.0 天然多样）。**用显式
   repeat 实现**（B prompt → 各发 n 次单条 generate = B×n 条），**不用 sglang 原生 `n>1`**——因 DSPARK 投机解码
   + `n>1` 未证实（无明确支持也无明确禁止），而 B×n 单条请求 = T3a 已验证的路径、保证每条独立、server continuous
   batching 下开销等价。
2. **喂变长不 padding、拿回自己 pad（对齐标准 verl）**：核实标准 verl `sglang_rollout.py:640-800`——喂 sglang 的是
   **变长 prompt token list（不 padding，引擎自组批）**；拿回**变长 response 后 verl 自己 `pad_sequence` 右填充**
   （`:223/765`）。故 T3b：prompt 传变长 list（不 pad），response 拿回后用 `dspark_collate_fn` 思路右填充重建 batch。
3. **max_new_tokens=2048（对齐金标准 §2.4）**：先对齐 evaluate 金标准跑通；若太慢/太占显存再降（延后项）。

**做法**（改 `recipe/dspark_opd/worker.py`，依赖 T3a 的 server 已起）：

**① worker 当 client（一次，`init_model` 尾部或首个 step 惰性）**：从 driver 拿到的 server `host:port` 构造
**client 句柄** `HttpServerEngineAdapter(host=..., port=..., launch_server=False)`（`launch_server=False` = 纯
HTTP client、不再起进程，`http_server_engine.py:262`）。8 个训练 worker 各持一个 client 指向同一个 server。

**★ 错峰策略已定（用户拍板 2026-07-17）：小 KV 池（`mem_fraction_static=0.15`）+ engine 常驻、不 release/resume。**
- **为何小池**：训练 rollout 的 KV 需求极小——tp=8 分片后 Qwen3-4B 仅 **0.018 MB/token/卡**，并发 16 条 @2048 也
  只 ~0.6GB/卡。**`mem_fraction_static=0.6` 是给在线 serving 的、训练用不着**。0.15（预留 12GB，扣权重~1.4+图杂~5
  ≈ **KV 池 ~9GB**，够并发数百条）留了安全垫（比 0.1 稳）。
- **为何常驻不错峰**：小池下 rollout 态每卡从 ~53GB（0.6）降到 **~16GB**（0.15），与训练态 ~32GB 共存 ≈ **~48GB
  < 80**，余量足。既然放得下，就不必每步 release/resume——**避开 T3a 实测的每轮 ~5s**（其大头是 cuda-graph 重映射
  + KV 池重分配 + cpu_backup 权重拷贝，非纯搬运；纯 11GB 权重 tp=8 分片后每卡仅 ~1.4GB、8 卡并行 ~0.5s）**+ 完全绕开
  cpu_backup 权重损坏风险**（§5.6a 那个坑根本不触发）。`mem_fraction_static` 做成可配，T3a smoke `--sim-train-gib 32`
  实测共存峰值坐实。
- **[后手] 若未来 batch/序列涨到共存放不下**：改 `release(tags=["kv_cache"])`——**只吐（本就不大的）KV 池、不动
  weights/cuda_graph**，故既无 cpu_backup 权重拷贝、也无 cuda-graph 重映射，比"全 release"快得多、且不碰权重（无损坏
  风险）。这是明确的降级路径，现在不做（小池常驻本就够）。

**② 注入点：融合 train_step 的 Phase-1（`worker.py:345`）**——把"draft nn.Module block 采样"
（`block_rollout.py:32,92`）替换为 sglang 现场投机解码（**常驻，无 resume/release**）：
```python
# ===== Phase 1: ROLLOUT（sglang DSPARK 现场投机解码；engine 常驻、不 release/resume）=====
# 每 rank 只处理自己的 DP 分片（NO_SHARD：train_step 已按 dp chunk 到各 rank）
prompts = [ii[:first_nonzero(lm)] for ii, lm in zip(b["input_ids"], b["loss_mask"])]  # §5.6a prompt 抽取
outs = [self._sglang_client.generate(input_ids=p, sampling_params=SP) for p in prompts]  # engine 常驻，直接打
responses     = [o["output_ids"] for o in outs]
accept_states = [o["meta_info"]["dspark_accept_state"] for o in outs]   # T1 流（T4 消费）
# 新序列重建：新 response → 重建 input_ids/loss_mask（新 resp 段=1）
#            → recompute_target_hidden_states（co-resident teacher，recompute 模式现成，teacher_scoring.py:33）
```
- `SP` 对齐金标准（§2.4）：`temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, max_new_tokens=2048`。
- **无 resume/release**（常驻方案，见上方"错峰策略已定"）——engine 权重+小 KV 池全程驻卡；后手降级路径见上。
- **每 rank 各发自己分片**：NO_SHARD 下 `train_step` 已把 batch chunk 到各 rank，各 rank 对同一个 tp=8 server
  发自己那份 prompt、收自己的 response（server 天然并发多 client）——**无需 rank0 gather/scatter**。
- **Phase-2 teacher / Phase-3 update 两段保持不动**。

**③ block plan 的过渡（T3b 可先于 T4/T5 独立测）**：新序列重建后，block plan **暂用旧的随机
`sample_anchor_positions`**（在**新 response** 上采 anchor）当占位——这样 T3b 不依赖 T4/T5 即可跑通端到端
（新 response → 合法训练张量 → teacher+update）。等 T5 就绪，再把 block plan 换成"用 accept_state 流重建
（anchor = seed + 各 boundary）"。**这是 T3b 与 T4/T5 的清晰接缝。**

**④ `DSparkRollout` 占位**：`generate_sequences` 仍不进主路径（融合 train_step 直接调 client），维持 registry
合规即可。

**落地文件（✅ 已实现，静态验证通过，待 GPU smoke）**：
- `recipe/dspark_opd/sglang_rollout_bridge.py`（纯函数，CPU 单测已过）：`extract_prompts`（`input_ids[:first
  loss_mask==1]`）、`sglang_generate_batch`（B×n **显式 repeat** 单条请求）、`rebuild_padded_batch`（变长
  response 右填充重建 input_ids/loss_mask/attention_mask）。
- `recipe/dspark_opd/worker.py`：`set_sglang_server(host,port)`（`ONE_TO_ALL`，各 worker 建
  `HttpServerEngineAdapter(launch_server=False)` client）+ `train_step` 内 `DSPARK_SGLANG_ROLLOUT=1` 门控的
  sglang 段（抽 prompt→B×n generate→重建→`recompute_target_hidden_states`，再走不变的 Phase-1/2/3）。
- `recipe/dspark_opd/trainer.py`：`_maybe_start_sglang_server`（`fit()` 内建 tp=8 actor + 播地址给 workers，
  `DSPARK_SGLANG_MEM_FRACTION` 默认 0.15）+ `_shutdown_sglang_server`（fit 末 + atexit 兜底）。
- **零回归**：flag 未设时走原 cache-response 路；纯训练 env（无 sglang）import 无碍，sglang 仅在
  `set_sglang_server` 内惰性 import。

**T3b 单独测试方案**（依赖 T3a；用旧随机 anchor 占位，不依赖 T4/T5/T6）：
- `scripts/opd/s_m2_t3b_rollout.py`（✅ 已写，bridge CPU 单测已过）：起 T3a server + 加载真 cache 样本 → 走
  bridge（extract→B×n generate→rebuild），断言：① B×n 行、每行 loss_mask 恰盖新 response 段、prompt 前缀保留；
  ② response 非空、多数 EOS(stop)；③ 同 prompt 的 n 条 rollout **互不相同**（temp=1.0 独立性）；④ 形状/dtype
  合 tensor-contract（§S1）。**期望末行 `RESULT: T3b ROLLOUT WIRING OK`。**（注：此 smoke 隔离测 bridge，不起
  完整 FSDP 训练循环；train_step 端到端在集成 §5.10 验。）
- **可选 `--check-forward`**：对新序列 `recompute_target_hidden_states`，证新序列能喂 recompute 路径。
- **量级对拍**：sglang 新 response 的 accept length 量级与 §4.2 golden 一致（temp=1.0 只比量级）。

### 5.7 任务 T4 `[V]` — draft 信息传输（generate dict `meta_info` → 训练侧张量）✅ 已实现（CPU 单测过）

> **⚠️ 机制简化（随 T3 定稿）**：T3 直驱 `HttpServerEngineAdapter`，`engine.generate(...)` 直接返回 raw
> server JSON dict，`meta_info["dspark_accept_state"]` 就是 T1 的流。故 T4 **不经 verl `ServerAdapter` 的
> `extra_fields` 搬运层、无需仿 `async_sglang_server.py`**——直接从 dict 取。**且 T3b 做法 X 下 `<think></think>`
> 壳在 prompt 里，`output_ids` 是壳后的纯答案，accept_state 恰好 response-indexed、`len == len(output_ids)`。**

**做法（✅ 落地，`sglang_rollout_bridge.py` + `worker.py`）**：
- `rebuild_padded_batch` 顺带抽每条 `meta_info["dspark_accept_state"]`（`_accept_state_of`），产出两个新张量：
  - `dspark_accept_state [B×n, R_max]` long——**response-indexed**（index j = response token j = 全局位
    `prompt_len+j`），右填充 **`ACCEPT_STATE_PAD=-1`**（因码 0=ACCEPT 是有效值，pad 必须用别的哨兵）。
  - `response_lengths [B×n]` long——每行 `len(output_ids)`，T5 用它切出 `[i, :rlen[i]]` 的干净流喂
    `dspark_stream_unit.reconstruct`。
- `worker.train_step` 的 sglang 段把这两个张量塞进 `b`（`dspark_accept_state`/`response_lengths`），供 T5 消费。
  **anchor 不传输**——T5 从 seed+boundary 推（§5.1）。
- 短/legacy 流（无 seed、len=resp−1）：`min(len(st), lr)` 截断保持 response 对齐，长度校验在 smoke 响亮报出。

**T4 单独测试方案（✅ CPU 单测通过）**：
- **单测（mock，无 server）**：mock 带 `dspark_accept_state`（首元素 SEED=2）+ `output_ids` 的 generate dict →
  `rebuild_padded_batch` → 断言 `dspark_accept_state` response-indexed、pad=-1、SEED@0、`len==response_length`、
  内容一致。**已过**（row0 `[2,1,-1]`、row2 `[2,0,1]`、rlen `[2,1,3,1]`）。
- **接真 server（`s_m2_t3b_rollout.py` 已内置 [T4 accept_state] 断言）**：跑真 prompt，断言每行 stream
  `len==response_length`、SEED@0、有效span 无 pad；缺流则 FAIL（提示 server 缺 T1 patch）。随 T3b GPU smoke 一起验。

### 5.8 任务 T5 `[V]` — draft 信息消费（去 Phase-1 + accept_state 重建 block plan + full-vocab 前向）✅ 已实现（CPU 端到端过）

> **落地文件（✅ CPU 单测+端到端过，待 GPU smoke）**：
> - `block_plan_reconstruct.py`（新，纯函数 CPU 测过）：`reconstruct_block_plan` = `_reconstruct_one`（复用
>   `dspark_stream_unit.reconstruct` 逻辑）→ 铺 `anchor_positions/block_keep_mask/eval_mask/tokens`。eval_mask
>   只标 accept 段（`n_acc=block_lengths-1`），boundary/reject 位 + 丢弃段 =0（选项 B）。tokens 填 response 真实
>   token。CPU 测：stop-trim、跨轮、reject 位排除 全对。
> - `worker.train_step`：`DSPARK_SGLANG_ROLLOUT=1` 且有 accept_state 时走 T5 分支——删 Phase-1、
>   `reconstruct_block_plan` 出 plan、Phase-2 teacher 返 top-64、Phase-3 draft 定 top-16 + align teacher；
>   否则原 cache-response 路（random anchor+top-k）不变。
> - `teacher_scoring.score_blocks_flat(teacher_top_k=64)`：返 `{t_ids,t_logp}` [B,A,blk,64]（logsumexp 真 logπ，
>   不物化 full-V log_softmax）。cache 路仍 `student_top_k_ids=<ids>` gather。
> - `loss_bridge.draft_topk_logp`（draft top-16 真 logπ 带梯度）+ `align_teacher_to_draft`（min-fill 对齐）。
> - Phase-3：`d_ids,S_grad=draft_topk_logp(...)`；`T=align_teacher_to_draft(d_ids,t64_ids,t64_logp)`；S/T 都
>   `[B,A,blk,16]` index-aligned → 喂 loss。
> - `losses.py`：`DSparkLossContext.S_grad/T_on_S` 恢复为 `[B,A,blk,K]`（删 full-vocab 分支/`_select_topk_ST`）；
>   `S_logp_old` 可 None（T5 下 confidence 目标从 `S_grad.detach()`/`T_on_S` 导）。**loss 数学一字未改。**
> - CPU 验证：draft logπ==真 full-vocab（logsumexp）、min-fill 逐 id 正确、grad 通、reverse-KL+confidence loss 有限；
>   cache 路回归不变；纯训练 env import 零回归。
> - **✅ GPU 数据路验证（2026-07-18，`s_m2_t3b_rollout.py`，8卡真 sglang output）**：T3b+T4+T5 全绿——
>   `[T4] streams=8/8 OK`、`[T5] max_anchors 动态=126（无截断，修了上次 max_anchors=64 截断 vc=91 的 FAIL）`、
>   `[T5 block plan] rounds==spec_verify_ct OK`（anchors 递增+in-range）、accept_len 5.11、`RESULT: T3b ROLLOUT
>   WIRING OK`。候选算法+完整 loss/前向留集成 §5.10 验。


> **★ 架构定稿（用户拍板 2026-07-17，多轮讨论后）——T5 是 Stage-M2 最大的一处结构改动，把融合
> `train_step` 从"随机 anchor 采样 rollout + top-k"改成"真实投机轨迹重建 + full-vocab"。七条已锁决策：**
>
> 1. **去掉 Phase-1（draft 采样前向）**：现 Phase-1（`dspark_block_rollout`）唯一"必须前置"的独有产物是
>    **top-k 候选 id**（Phase-2 打分前要知道候选）。**改 full-vocab 后候选前置需求消失** → Phase-1 整个删除。
>    其余产物（tokens/anchor/keep/eval_mask）本就该由 T5 从 accept_state+response 重建，非 Phase-1 采样。
>    **⇒ 前向 3 次降 2 次**（teacher 打分 + draft 带梯度）。
> 2. **候选选择算法（用户 2026-07-18 修订，替代原"返 full-vocab 到 loss"——那个 A 大时 `[B,A,blk,V]` OOM）**：
>    每 block —— **teacher 取 top-64**（`score_blocks_flat(teacher_top_k=64)`，真 full-vocab logπ = `logits.topk(64)
>    − logsumexp(V)`，**logsumexp 标量归约、不物化 `[B,A,blk,V]` log_softmax**，避开显存点）；**draft 取 top-16**
>    （`draft_topk_logp`，`only_stu` 候选、同法真 logπ 带梯度）= loss 支撑集；**teacher 在 draft 的 16 个 id 上取值**
>    （`align_teacher_to_draft`）：在 teacher top-64 内→真 logπ，不在→`min(teacher top-64)`（下界代替）。**loss 数学
>    不变**（仍在 draft top-K=16 上，K 从 config `log_prob_top_k`）。teacher 从此不算 full-V log_softmax；draft 的
>    full-V log_softmax 图也由 logsumexp 省掉一份 output。`dspark_teacher_top_k` 默认 64（config 可调）。CPU 实测：
>    draft logπ == 真 full-vocab、min-fill 逐 id 正确、grad 通、loss 有限。
> 3. **block plan 从 accept_state 重建**：复用 `scripts/opd/dspark_stream_unit.py:reconstruct`（T1 单测已验，含
>    stop-trim）——`reconstruct(accept_state) → (anchors, block_lengths)`：`anchors[r]` = round r 的
>    **output_ids 索引**（r=0 是 SEED@0，其后 = 上一轮 boundary），`block_lengths[r]` = 该轮提交数。**删掉** fork
>    大集方案的反推 anchor / `min(block_size-1)` 截断 / 补块 / 去重（`dflash_student.py:484,519-611`）。
> 4. **block 内 token = response 真实 accept token（真实轨迹，非重采）**：block r 铺
>    `[anchor | accept_1..accept_{L_r}]`，token 取自 response 里该 round 的真实接受 token（`block_prev_tokens`
>    错位一格填法天然正确，§markov 逐位独立）。draft 学"复现真实投机轨迹"而非随机位置重采。
> 5. **变长 block 用 eval_mask 表达，张量仍定长 `[B,A,block_size]`**：真实接受长度 L_r ≤ block_size；每 block
>    `eval_mask` 只标前 L_r 位（markov 并行前向逐位独立，变长零成本，不需 ragged tensor）。
> 6. **reject 位（boundary，第 L_r+1 位）= 选项 B**：**只记 q/p、不进当前 loss**（`eval_mask` 该位 = 0；loss
>    语义留后续实验改）。其 draft 分布 q 只依赖 anchor+前 L_r 个 accept token（全在 response），target 分布 p
>    由 teacher 对 `[anchor,accept_1..accept_L]` 前缀打分——**q/p 都靠重算、逐位对齐真实轨迹，不用 sglang 钩子**。
> 7. **丢弃段（L_r+2 起）不覆盖**：draft 在被拒位提议的 token 没传出、重算不可复现；但超出选项 B 范围，`eval_mask`
>    该段 = 0。（未来若要研究被拒分支，走"钩子导被拒 token id"独立任务，非 T5。）

**做法（改 `recipe/dspark_opd/`）**：
- **`worker.train_step` 的 sglang 段（§5.6b）**：新序列重建后，用 `b["dspark_accept_state"]`（T4，response-indexed）
  逐样本 `reconstruct` → 得 `(anchors_output_idx, block_lengths)` → 转成 `anchor_positions [B,A]`（=
  prompt_len + anchor 的 output_idx）、`block_keep_mask [B,A]`、`eval_mask [B,A,block_size]`（每 block 前 L_r 位
  =1，reject/丢弃位 =0）、`tokens [B,A,block_size]`（response 真实 token 铺入）。A = max round 数（不足补 keep=0）。
- **删 Phase-1**：`train_step` 不再调 `dspark_block_rollout`；`anchors/keep/eval_mask/tokens` 全来自上面的重建。
- **Phase-2 `score_blocks_flat`**：删末尾 `gather(student_top_k_ids)`（`teacher_scoring.py:179`），直接返回
  `logp [B,A,blk,V]`（`T_on_S_full`）。入参不再需要 `student_top_k_ids`。
- **Phase-3 `module(...)`**：`out.draft_logits [B,A,blk,V]` 直接 `log_softmax` 出 `S_grad_full [B,A,blk,V]`（删
  `logp_on_topk_ids` 的 gather），带梯度。
- **loss（`losses.py`）**：`DSparkLossContext` 的 `S_grad/T_on_S` 从 `[B,A,blk,K]` 变 `[B,A,blk,V]`；reverse_kl
  operator 现取 top-k（或按实验换选法）——**这是 loss 阶段的事，T5 只保证前向出 full-vocab + 正确 block plan**。
- reject 位 q/p：从 full-vocab 的 `S_grad_full`/`T_on_S_full` 在 reject 位（每 block 第 L_r 位，即 boundary 前
  一位对应的 draft 提议位）取——**记录供实验，不进 loss**（选项 B）。

**T5 单独测试方案**：
- **CPU 端到端（无需 GPU）**：喂已知 accept_state 流 → `reconstruct` + 铺 block plan → 断言 anchor/block_lengths
  与手算一致（含 stop-trim、跨多轮整块接受、reject 位 eval_mask=0）；对照 eval 侧 `draft_ops.py` 的 slot 语义
  （slot i→anchor+1+i）。
- **小 GPU 验证 full-vocab 前向**：给一条真实 response+accept_state → 删 Phase-1 后跑 Phase-2/3 → 断言
  `T_on_S_full/S_grad_full` shape `[B,A,blk,V]`、数值有限、accept 段+reject 位都取到、带梯度 backward 正常。
- **量级不变式**：重建的 accept length（Σblock_lengths/round 数）== meta_info `spec_accept_length`。

### 5.9 任务 T6 `[V]` — draft 权重同步接线（每 step 推 draft 回 sglang）✅ 已实现（待 GPU 集成验）

**做法（`recipe/dspark_opd/worker.py`，`_push_draft_weights_to_sglang`）**：
- **经 worker 的 sglang client 打回 server**：worker 的 `self._sglang_client`（T3b 建的
  `HttpServerEngineAdapter(launch_server=False)` HTTP client）调 `update_weights_from_tensor(named,
  draft_model_only=True, flush_cache=True)`——HTTP 只传 base64 IPC meta，权重本体 **GPU 直拷（CUDA-IPC）跨
  "训练 worker 进程 → server 子进程"、同物理卡**。**这正是 T2 probe 验证过的同一条调用**
  （`dspark_draft_update_probe.py:220/362`：draft 变/target 冻结/round-trip）。（用 client 直调，非经 driver 的
  actor `.remote()`——worker 持的是 client 不是 actor 句柄。）
- **权重导出（复用 verl rollout_mode 惯用法 `fsdp_workers.py:674-709`）**：`actor_module.state_dict()` →
  `convert_weight_keys`（剥 FSDP 前缀）→ DTensor 则 `.full_tensor()`。丢 `embed_tokens./lm_head./rotary_emb.`
  （冻结、`Qwen3DSparkModel.load_weights` 本就跳过 = §4.0 的 47/47 CLEAN 集）。**权重名无 `draft_model.` 前缀**
  ——那是 Draft-OPD 的 composed-student 包装，我们的 `actor_module` 直接就是 `Qwen3DSparkModel`，键名已与
  `load_weights` 逐字匹配，**无需剥前缀**。
- **⚠️ tp_size 关键坑**：`update_weights_from_tensor` 按 `server_args.tp_size` **每 TP rank serialize 一份**
  （server 把 `serialized_named_tensors[rank]` 分给各 TP worker，`io_struct.py:1596`）。裸 `host/port` 建的 client
  默认 `tp_size=1` → 只发 1 份 → rank 1..7 收不到 → draft 同步残缺。故 `set_sglang_server` 收 `tp_size` 并设到
  `client.server_args.tp_size`（=8），保证发 8 份。
- **谁推 + 时序**：**仅 rank0** 在融合 `train_step` 的 update（Phase-3）之后推（NO_SHARD 下各 rank draft 相同，推
  rank0 即可）；其余 rank `barrier` 等（下一轮 rollout 前 draft 已同步）。`flush_cache=True` 清 radix/KV，保证下一轮
  用新 draft 重算。返回 `actor/draft_weights_pushed` 指标（0 → raise 护栏，防静默空导出）。

**T6 单独测试方案**：
- **底层通路 T2 已证**（`dspark_draft_update_probe.py`：`update_weights_from_tensor(draft_model_only=True)` →
  draft.* checksum 变 / target 冻结 / round-trip 精确）。T6 只是把"从 disk 读扰动张量"换成"从训练侧真实
  `actor_module` state_dict 剥冻结件导出"，通路同一条。
- **集成验（§5.10）**：多 step 闭环里断言 `draft_weights_pushed>0`、且 sync 后 server 侧 draft checksum ==
  训练侧当前 draft、target 全程不变、accept length 随训练变化（跟随 draft 更新）而非恒定初值。
- 量级护栏：`_push_draft_weights_to_sglang` 已内置"导出 0 张量直接 raise"。

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
5. **共卡显存/时序稳定（关键）**：多 step 下 rollout 态峰值稳定 ≤ 80GB（不随 step 累积泄漏）；每 step
   resume→generate→release→update→推权重（`draft_model_only`）的时序不死锁、release 后训练态显存正常回落。

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
| `s_m2_t3a_server.py`（待建）| T3a：driver 建 tp=8 独立 server actor（共驻 8 卡）→ 金标准口径发 prompt，验 generate/accept_state/release-resume（借 `dspark_draft_update_probe.py:312` engine 构造）| Stage-M2 T3a（§5.6a）|
| `s_m2_t3b_rollout.py`（待建）| T3b：起 T3a server + 融合 train_step Phase-1 走 sglang，验新 response→重建张量→旧随机 anchor 铺 block→teacher+update 不崩（不依赖 T4/T5）| Stage-M2 T3b（§5.6b）|
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
| verl async server 编排壳（**不复用**，仅参照）| `replica.py:146`（`init_hybrid`）、`agent_loop.py`（`AgentLoopManager`）、`fsdp_workers.py:2762`（`AsyncActorRolloutRefWorker.wake_up/sleep`，仅 async 有）|
| **T3a 采用：直驱 sglang 引擎句柄** | `verl/workers/rollout/sglang_rollout/http_server_engine.py:194`（`HttpServerEngineAdapter`）：`:78` `update_weights_from_tensor(draft_model_only)`、`:109` `generate`、`:143-147` `release/resume_memory_occupation`、`:137` `multiprocessing.Process` 起子进程、`:262` `launch_server=False` = 纯 client |
| **T3a server actor 物理放置法（复刻 verl async）** | `async_sglang_server.py:261-278`（收 worker `(node_id, CUDA_VISIBLE_DEVICES)`）、`:284-300`（`NodeAffinitySchedulingStrategy(soft=False)` + `runtime_env RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`）|
| **⚠️ 训练 worker 只见 1 卡（tp=8 须独立 actor 的根因）** | `single_controller/base/worker.py:248-256`（Ray 逐 actor 设 `CUDA_VISIBLE_DEVICES`）、`ray/base.py:217,380`（`num_gpus=1`/actor）；recipe 未设 NOSET |
| **T3a 最小工作范例（T2 已验同一句柄）** | `scripts/opd/dspark_draft_update_probe.py:312`（构造 `HttpServerEngineAdapter(DSPARK)` + generate + `update_weights_from_tensor(draft_model_only)`）|
| CUDA-IPC 权重同步跨进程同物理卡可用 | `sglang/srt/utils/cuda_ipc_transport_utils.py:130`（`_share_cuda_` handle）；T6 训练 worker→server actor 靠此 |
| verl 进程内 sync 丢 spec 参数（排除依据）| `sglang_rollout.py:440-472`（`_init_inference_engine` 硬编码 whitelist、不 splat `engine_kwargs`）|
| verl 无 Worker teardown 钩子（T3a 生命周期须自管）| `single_controller/base/worker.py`（无 `__del__/shutdown/close`）；T3a 靠 `DSparkTrainer.fit()` 末尾显式 shutdown + `atexit` 兜底 |
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
