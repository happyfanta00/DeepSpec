# DSpark On-Policy Distillation (OPD) 实验设计文档

> 本文档用于设计并记录对 DSpark 草稿模型（draft model）进行 **On-Policy Distillation (OPD)** 训练的实验方案。文档将逐步完善。
>
> - **第 1 部分（已完成）**：学习并总结「什么是标准 LLM-OPD」，作为后续设计的知识基础。参考代码库：`/home/ec2-user/efs_data/workspace/Rethink-OPD/`（清华 thunlp 的 *Rethinking On-Policy Distillation*，基于 verl v0.7.0 的 fork）。
> - **第 2 部分（已完成）**：DSpark 与标准 OPD 的差异分析、DSpark-OPD 训练方案细化与可行性验证。已锁定两项关键决策——Teacher 打分走**路线 A（online target 打分）**、框架用 **verl 0.7.0**（Rethink-OPD fork 版本；曾评估的 0.8.0 因实现过重转为备选，见附录 [`DSpark-OPD-verl080.md`](./DSpark-OPD-verl080.md)）。
> - **第 3 部分（进行中）**：开发计划——8 个阶段（S0 环境+verl 骨架 → S7 eval 对照），**verl 从 S0 接入、四个集成点先占位桩、沿 dataflow 逐个变真**；每阶段两层测试：**smoke-test（我自动跑）** + **E2E test（你用 verl 入口启动、`stop_after` gate 打印真实中间输出并 `exit(0)`）**。**进度：S0 ✅ / S1 ✅ / S2 ✅（gate=`rollout` rc=0）/ S3 ✅（teacher 块对角打分，展平等价性黄金校验通过）/ S4 ✅（reward→adv→PG+confidence loss→backward；多卡梯度同步双层修复：训练前向走 `FSDP.forward` + actor 改 `NO_SHARD`；单卡 smoke A–G + 多卡 grad-sync smoke `Δparam=0` 通过）/ S5 ✅（fit 架构：自建多 step 循环，非 super().fit()）/ S6 ✅（多卡多 step：8 卡 200-step 完整训练跑通，checkpoint 落 `s6_run/global_step_{100,200}`；数据流经 #3/#4/#5 优化，每 step 56.6s→8.5s）/ S7 ✅（`scripts/opd/convert_ckpt.py` verl FSDP→HF safetensors，64 张量 bit-exact；转换+eval 链路冒烟通过 gsm8k×5 `acceptance_length=6.01`；**完整 eval.sh 对照基线待你执行**）**。
>
> **⚠️【架构演进：融合 + 去 gate（S6 性能优化，最新）】** py-spy + 计时实测发现每 step ~75-80% 时间耗在 Ray object-store 传输（`target_hidden_states` 400MB 在 rollout/update 各 dispatch 一次 + rollout/teacher 中间产物 collect 回 driver 再 re-dispatch）。**已把 rollout→teacher→update 三次独立 worker RPC 融合为一次** `actor_rollout_wg.train_step(batch)`：worker 内顺序跑完整步、中间产物留显存、只回标量 metrics。连带：① **teacher 并入 actor worker**（`_build_teacher`，取消独立 RewardModel role/rm_wg）；② **移除 stage-gate 机制**（`gate.py` 删除、`stop_after`/`STOP_AFTER`/`dump` 去掉——S0-S5 已验证，gate 开发脚手架完成使命）。**下文 S1-S5 里 `STOP_AFTER=xxx` 的 gated E2E 命令是历史开发记录**（当时逐阶段验证用），现已不适用；当前训练/E2E 统一用 `bash run.sh`（见 §S6 / run.sh）。融合的正确性由 `fused_step_smoke.py`（融合 per-micro 切分 == 整批梯度）+ 保留的 s3/s4/grad_sync smoke（底层纯函数）共同保证。详见 `docs/opd/fused-step-design.md`。
>
> **⚠️【数据流优化 #3/#4/#5（融合之后，最新）】** 融合后 py-spy 定位到剩余瓶颈是 driver 单点 dispatch/序列化 `target_hidden_states`。三步演进（step 均值 56.6s→8.5s，8 卡 200-step 实测）：**#3** rollout.n 的 repeat 从 driver 挪到 worker（去 4× 冗余副本，`DSPARK_REPEAT_ON_DRIVER` A/B 开关）；**#4** worker 按 index 从 cache 读 hidden（暴露磁盘冷读，非默认）；**#5（现默认，`DSPARK_HIDDEN_MODE=recompute`）** driver 只 dispatch token、worker 用常驻 teacher 重算 `target_hidden_states`（一次 prefill forward+hooks）。**故下文 §2.2/§2.4/§2.9 里"`target_hidden_states` 从 cache 经 dispatch 传给 worker"的数据流叙述，现仅适用于 legacy `dispatch` 模式；默认 recompute 模式下由 worker 重算（张量 shape/contract 不变，只是来源变，且与推理的实时 teacher forward 对齐）。** 详见 `docs/opd/worker-side-cache-read-design.md`。

---

## 第 1 部分：标准 LLM-OPD 剖析（基于 Rethink-OPD）

### 1.1 OPD 是什么，为什么用它

**On-Policy Distillation（在线策略蒸馏）** 是一种后训练（post-training）方法，核心思想：

- 让 **Student（学生 / 待训练策略模型）** 自己生成回复（rollout），
- 用 **Teacher（教师模型）** 对 Student **自己走过的 token 轨迹** 逐 token 打分，给出「教师认为该位置应该怎么分布」的稠密信号，
- Student 朝教师分布对齐。

它介于两类方法之间，兼取两者之长：

| 方法 | 数据分布 | 监督信号 | 主要问题 |
|---|---|---|---|
| **离线蒸馏 / SFT-on-teacher** | 教师生成的轨迹（off-policy） | 稠密（每 token 教师分布） | 训练/推理分布不匹配（exposure bias）；学生看不到自己的错误状态 |
| **RL (GRPO/PPO)** | 学生自己生成（on-policy） | 稀疏（序列级 reward，通常只有对/错） | 信用分配难、样本效率低 |
| **OPD** | **学生自己生成（on-policy）** | **稠密（每 token 教师分布）** | 需要教师在线前向，算力开销大；能否 scale 到长程仍存疑 |

一句话：**OPD = on-policy 的数据分布 + 稠密的 token 级教师监督**。在学生「实际会到达的状态」上，用教师给出密集反馈，因此既避免了 exposure bias，又比 RL 的稀疏 reward 信息量高得多（论文称之为「dense token-level reward 的免费午餐」，同时指出这份午餐在长程蒸馏上有代价）。

论文的两个关键结论（对我们选 Student/Teacher 有直接指导意义）：

1. **思维模式要兼容**：Student 与 Teacher 的「思考模式」需相容，否则 OPD 失败；
2. **Teacher 必须提供 Student 训练中「没见过的新能力」**：同族的小/大模型（如 1.5B 与 7B 同族）从学生视角看分布上「无法区分」，反向蒸馏（weak-to-strong）会失效。成功的 OPD 表现为「在学生访问过的状态上，对高概率 token 的逐步对齐」，且大部分概率质量集中在一个很小的共享 token 集合（97%–99%）。

> 这直接引出了 OPD 的一个工程核心：**只需在每个位置的少数几个候选 token（top-k）上对齐**即可，不必对整个词表算 KL —— 这也是下面「top-k reward」的动机。

---

### 1.2 系统总体结构（Rethink-OPD 的实现范式）

Rethink-OPD 把 OPD **完全实现在 verl 的 RL 训练框架里**，几乎不引入新的 loss，而是通过「自定义 reward + 自定义 advantage estimator」复用 PPO 的策略梯度管线。关键映射：

- **Teacher = verl 里的 `reward_model` worker**（`REWARD_MODEL_PATH`），它 **不生成**，只对学生 token 逐位置前向、给出 log-prob 派生的 token 级 reward。
- **Student = verl 里的 actor**（`ACTOR_MODEL_PATH`），负责 rollout（vLLM）+ 训练。
- **Advantage estimator = `token_reward_direct`**：直接把 token 级 reward 当 advantage，不做 GAE/GRPO 归一。
- **Loss = 标准 vanilla PPO 双裁剪 surrogate**，扩展支持 3D `(batch, seq, k)` 的 top-k advantage。**没有专门的蒸馏 loss** —— 蒸馏信号全部藏在 reward/advantage 里。

启动入口：`python3 -m verl.trainer.main_ppo ...`（见 `on_policy_distillation.sh`）。每个训练 step 的编排（`verl/verl/trainer/ppo/ray_trainer.py:1042-1144`）：

```
1. generate_sequences()          # 学生用 vLLM 生成 n 个回复（on-policy rollout）
2. compute_log_prob()            # 学生再前向：产出 student_top_k_ids / student_top_k_log_probs
3. compute_rm_score()            # 教师前向：在学生 token 上产出 teacher log-probs（对齐到学生轨迹）
4. compute_distillation_reward() # 组装 token 级 reward = 加权的 (teacher_logp - student_logp)
5. compute_advantage()           # token_reward_direct: advantage = reward * response_mask
6. update_policy()               # 学生用 dual-clip PPO 对 top-k 候选做策略梯度更新
```

> ⚠️ 注意：verl_example 目录下还有另一套 `opd.sh`，它走 `distillation.*` 配置（`loss_mode=k1 / forward_kl_topk`），是更「显式」的蒸馏封装。但仓库主推、README 主线用的是上面这套 `token_reward_direct` + reward_model 的范式，本节以它为准。

---

### 1.3 Student 模型选择

- 配置项：`ACTOR_MODEL_PATH`（`on_policy_distillation.sh:109`），如 `DeepSeek-R1-Distill-Qwen-1.5B`、`Qwen3-1.7B-SFT` 等。
- 典型是 **1.5B / 1.7B 量级**的小模型，通常已经过 SFT「冷启动」（见 §1.8）。
- Student 同时承担两个角色：**rollout 生成器**（vLLM 推理）和**被训练的策略**（FSDP 训练）。verl 的 `ActorRolloutRefWorker` 把二者放在同一组权重上。
- 论文强调 Student 与 Teacher 需**思维模式兼容**、且 Teacher 要带来 Student「未见过的能力」，因此常见搭配是「弱学生 ← 强教师」，而非同族强弱互蒸。

### 1.4 Teacher 模型选择

- 配置项：`REWARD_MODEL_PATH`（`on_policy_distillation.sh:130`），如 `JustRL-DeepSeek-1.5B`、`Qwen3-4B`、`Skywork-OR1-Math-7B` 等。
- Teacher 作为 verl 的 `reward_model` 挂载（`reward_model.enable=True`），**只做前向打分，不更新、不生成**。
- **词表约束**：Teacher 与 Student 必须共享 tokenizer / 词表（reward 是在同一批 token id 上对齐 log-prob 的）。若词表不同，需要 `_switch_chat_template_token_level` 重新分词对齐（`fsdp_workers.py:2370`），但仍要求词表可映射。
- **Teacher 通常比 Student 强**（更大、或经过更强 RL / 数学微调），以满足「提供新能力」的条件。也可用「同尺寸但更强训练」的模型（如 1.5B student ← 1.5B JustRL teacher）。

### 1.5 输入数据格式

训练数据是标准 verl RL parquet（如 `datasets/dapo-math-17k.parquet`，17,917 行），**每行只含 prompt 与答案元信息，不含教师回复**（回复是 on-policy 现场生成的）。5 个顶层字段：

| 字段 | 类型 | 说明 / 示例 |
|---|---|---|
| `data_source` | string | `math_dapo` / `DeepMath-103K` 等，决定用哪个 reward 验证器 |
| `prompt` | list<struct{content, role}> | 聊天格式消息列表，如 `[{"role":"user","content":"Solve ... Answer:"}]` |
| `ability` | string | 领域标签，如 `MATH` |
| `reward_model` | struct{ground_truth, style} | `{"ground_truth":"34","style":"rule-lighteval/MATH_v2"}` |
| `extra_info` | struct{index, ...} | 索引 / 原始问题等 |

要点：

- **`reward_model.ground_truth`（标准答案）在 OPD 中不参与 token 级蒸馏 reward**，只用于日志里的 `true_reward_score`（可选给 GRPO-outcome 混合项用）。OPD 的训练信号完全来自教师 log-prob。
- 数据以数学推理为主（DAPO-Math-17k、OpenThoughts3、DeepMath-103K 等）。
- **Chat template**：`RLHFDataset` 用 `tokenizer.apply_chat_template(messages, add_generation_prompt=True, **apply_chat_template_kwargs)`。非思考模型需加 `+data.apply_chat_template_kwargs.enable_thinking=False`。
- **长度限制**：`max_prompt_length=1024`、`max_response_length=7168`；`filter_overlong_prompts=True` 先过滤超长 prompt，`truncation='error'` 对残余超长直接报错（安全断言，不静默截断）。

### 1.6 Rollout 方法（on-policy 的关键）

- **引擎**：vLLM（`actor_rollout_ref.rollout.name=vllm`）。
- **每 prompt 生成 `n=N_RESPONSES=4` 个回复**；`ray_trainer.py:1042` 先把 prompt 重复 4 份（interleave），vLLM 内部 `n=1`。
- **采样参数**：`temperature=1.0`、`repetition_penalty=1.0`、`top_p=1`/`top_k=-1`（默认），`max_tokens=7168`。
- **on-policy 保证**：回复由**当前 step 的学生权重**现场生成，随后 `compute_log_prob` / `compute_distillation_reward` 再用**同一份学生权重**前向。这就是「on-policy」的来源 —— reward 与梯度都作用在学生此刻真实会产生的分布上。
- **验证 rollout** 参数不同：`val_kwargs.n=16, temperature=0.7, top_p=0.95`。（README 提示 verl v0.7.0 内建验证会低估 5–7 个百分点，建议 `test_freq=-1` 关掉，用 `scripts/val/` 单独评测。）

### 1.7 Loss / Reward 计算方法（OPD 的数学核心）

分三段：**教师打分 → 组装 token reward → 策略梯度 loss**。

#### (a) Token 级 reward = 加权的逐 token 反向 KL

在学生每个回复位置 `t`，取一个**候选 token 集合**（默认为学生分布的 top-k），对集合内每个候选 token `k` 计算：

```
rm_scores[t, k] = w_k · ( logπ_teacher(token_k) − logπ_student(token_k) )
```

- `logπ_student(token_k)`：学生对候选 token 的 log-prob（学生 logits 的 log-softmax，`dp_actor.py:271`）。
- `logπ_teacher(token_k)`：**教师在同一 token 上**的 log-prob（教师全词表 log-softmax 后 gather，`fsdp_workers.py:1891`，不截断）。
- `w_k`：归一化权重（见 `REWARD_WEIGHT_MODE`）。

这本质是 **`−KL(student ‖ teacher)`** 在 top-k 候选集上的加权分解：教师给某 token 的概率比学生高 → reward 为正 → 推动学生提高该 token 概率。默认实现（`only_stu` + `student_p`）：

```
rm_scores[t,k] = softmax_k(student_logp) · ( teacher_logp_on_student_token_k − student_logp_k )
```

代码：`dp_actor.py::compute_distillation_reward`（451-624），权重函数 `compute_reward_weights`（522-556）。

**候选集选择 `TOP_K_STRATEGY`（决定哪些 token 进入 KL 求和）：**

| 策略 | 候选集 | KL 项 | 备注 |
|---|---|---|---|
| `only_stu`（默认） | 学生 top-k | `S_logp − T_on_S` | 只需教师前向一次，最省算力 |
| `only_tch` | 教师 top-k | `S_on_T − T_logp` | 需学生额外前向算 `S_on_T` |
| `intersection` | 学生∩教师 top-k | 同 only_stu，非交集位置置 0 | 只对齐共享 token |
| `union` | 学生∪教师（2K 宽） | 拼接两侧 | 去重避免重复计数 |
| `union-intersection` | 对称差（各只在一侧） | 拼接，用非归一化权重 | |

**权重 `REWARD_WEIGHT_MODE`：**
- `student_p`（默认）：权重 ∝ 学生对候选 token 的概率（K 维 softmax 归一）；
- `teacher_p`：权重 ∝ 教师概率；
- `none`：均匀权重 `1/K`。

**`LOG_PROB_TOP_K`：** `=16` 走上面 3D `(batch, seq, K)` 稠密 top-k 路径；`=0` 退化为**只在采样 token 上**的单点估计 `rm_scores = teacher_logp(sampled) − student_logp(sampled)`（即 k1 单样本 `−KL`），reward 变 2D。

**`TEACHER_TEMPERATURE`：** 对教师 logits 先除以温度再算 log-softmax / top-k（`fsdp_workers.py:2015`），`=1.0` 不缩放；`<1` 锐化、`>1` 平滑教师分布。学生侧由采样 `temperature` 单独缩放，二者独立。

#### (b) Advantage：`token_reward_direct`

直接令 `advantage = returns = rm_scores * response_mask`，不做任何组内归一化（`core_algos.py:854-880`）。3D reward 时把 mask 广播到 `(batch, seq, 1)`。

#### (c) 策略梯度 loss：dual-clip PPO（复用，非蒸馏专用）

`compute_policy_loss_vanilla`（`core_algos.py:1058-1197`）的 3D 分支：

```
r_{t,k}    = exp(logπ_θ(y_{t,k}) − logπ_old(y_{t,k}))          # 重要性采样比
pg_losses1 = −A_{t,k} · r_{t,k}
pg_losses2 = −A_{t,k} · clip(r_{t,k}, 1−ε_lo, 1+ε_hi)          # PPO 裁剪 (ε≈0.2)
loss_{t,k} = max(pg_losses1, pg_losses2)                       # A<0 时再加 dual-clip 下界 (c=3.0)
L_t        = Σ_k loss_{t,k}                                    # 对 K 维求和 → (B,T)
```

关键机制：更新时**用带梯度的学生前向重新在「构造 reward 时那同一批 top-k token id」上 gather log-prob**（`dp_actor.py:826-841`），保证 loss 的 log-prob 与 advantage 的候选集严格对应（apples-to-apples）。

**on-policy 的体现**：当 `ppo_epochs=1` 且 mini-batch = 整个 batch 时，`old_log_prob = log_prob.detach()`，故 `r ≡ 1`，裁剪 surrogate 退化为
```
L ≈ − Σ_t Σ_k A_{t,k} · logπ_θ(y_{t,k})
```
即「用教师-学生 KL 差作权重的 top-k 加权策略梯度」——这正是 OPD 的目标函数。若开启 mini-batch 或 `ppo_epochs>1`，则回退到用存储的 old log-prob，裁剪生效（off-policy）。

**聚合 `LOSS_AGG_MODE`**（对 K 求和后的 `(B,T)` 再聚合，`agg_loss` `core_algos.py:942-978`）：`token-mean`（默认）/ `seq-mean-token-sum` / `seq-mean-token-mean` / `seq-mean-token-sum-norm`。

**可选 KL 正则 `USE_KL`**（默认关）：对**冻结 reference 模型**加 `kl_loss_coef · KL(π_θ ‖ π_ref)`，与教师蒸馏 reward 无关；注意它作用于 2D log-prob，在 3D top-k 配置下一般保持关闭。

最终：`loss = pg_loss (− entropy·coeff) (+ kl·coef)`，乘 `loss_scale_factor` 后 `backward()`。

---

### 1.8 SFT「冷启动」（off-policy 预热，可选前置步骤）

论文提出 OPD 失败时的补救之一是「off-policy cold start」：先用教师离线生成回复做一轮 SFT，再进 OPD。

- 脚本：`scripts/infer/vllm_rollout.py`，用**教师**离线批量生成回复，存成 ShareGPT 风格 JSONL 供 LlamaFactory SFT。
- **拒绝采样**（`--enable-rejection-sampling`，最多重试 `--max-attempts-per-rollout=3`）：过滤掉①无 `\boxed{}`、②整行重复、③100-char n-gram 重复、④长文本连续重复块 的退化生成。
- 这是 **off-policy** 的（教师轨迹），与主 OPD 的 on-policy rollout 相对。

---

### 1.9 标准 LLM-OPD 要素速查表

| 要素 | Rethink-OPD 的做法 | 关键配置 / 代码 |
|---|---|---|
| **框架** | verl v0.7.0（RL 管线复用） + vLLM rollout + FSDP 训练 | `main_ppo` |
| **Student** | 1.5B/1.7B 小模型（常先 SFT 冷启动），作 actor | `ACTOR_MODEL_PATH` |
| **Teacher** | 更强/同尺寸强训模型，作 `reward_model`，只前向打分，**共享词表** | `REWARD_MODEL_PATH` |
| **数据** | 标准 verl RL parquet（prompt+答案元信息，无教师回复） | `dapo-math-17k.parquet` |
| **Rollout** | on-policy，vLLM，每 prompt n=4，T=1.0 | `rollout.name=vllm, n=4` |
| **监督信号** | token 级、top-k 候选上的加权 `−KL(student‖teacher)` | `compute_distillation_reward` |
| **Advantage** | `token_reward_direct`：reward 直接当 advantage | `ADV_ESTIMATOR` |
| **Loss** | dual-clip PPO surrogate（on-policy 时 ≈ top-k 加权策略梯度） | `compute_policy_loss_vanilla` |
| **超参** | top_k=16, strategy=only_stu, weight=student_p, teacher_T=1.0, lr=1e-6 | `on_policy_distillation.sh` |

### 1.10 关键源码索引（Rethink-OPD）

- 启动脚本：`on_policy_distillation.sh`（主线）、`verl_example/opd.sh`（distillation.* 变体）
- Step 编排：`verl/verl/trainer/ppo/ray_trainer.py:1042-1144`
- Token reward 组装 + 5 种策略 + 权重：`verl/verl/workers/actor/dp_actor.py:451-624`
- 学生 top-k log-prob（带梯度 gather）：`verl/verl/workers/actor/dp_actor.py:203-271, 826-841`
- 教师前向 / 对齐 / 温度：`verl/verl/workers/fsdp_workers.py:1830-1917, 1993-2290, 2553-2758`
- Advantage estimator：`verl/verl/trainer/ppo/core_algos.py:854-924`
- 策略 loss（dual-clip PPO，3D 分支对 K 求和）：`verl/verl/trainer/ppo/core_algos.py:1058-1197`
- 数据集加载 / chat template / 过滤：`verl/verl/utils/dataset/rl_dataset.py`
- 离线教师 rollout（SFT 冷启动）：`scripts/infer/vllm_rollout.py`

---

## 第 2 部分：DSpark-OPD 设计

### 2.0 设计骨架（用户既定，本部分在此之上细化并验证可行性）

- **Student = DSpark draft model；Teacher = Target Model（Qwen3-4B 本身）。**
- **On-Policy 的含义**：基于 Target Model 生成的 Response（`scripts/data/start_generate_train_data.sh`）和随机选择的 anchor 位置，让 Draft Model 在这些 anchor 处做 **block generation 采样**得到 token 序列（即 rollout），而非像现在这样对着真实后继 token 做 teacher-forcing。
- **输入数据**：复用缓存的 Target hidden state（`scripts/data/start_prepare_target_cache.sh`）。
- **Loss**：采用标准 OPD loss，用 **Teacher Model 对 rollout 计算概率**。
- **框架**：使用 verl，最初提出基于 **verl 0.8.0**（利用其对 OPD 的原生支持）。注意 rollout 引擎选择——**vLLM 无法支持 Draft Model 的块并行 rollout 模式**，须自研基于 PyTorch 的 rollout，参考 `scripts/eval/eval.sh`（`deepspec/eval/dspark/`）中 Draft Model 的推理实现。〔**更新**：经可行性调研，0.8.0 实现过重，已改为 **verl 0.7.0**，见下方「两项关键决策」与 §2.6。〕
- **评估**：完全复用 `scripts/eval/eval.sh`，**不做任何修改**。

**两项关键决策（已锁定）**：
- **Teacher 打分 = 路线 A（online target 打分）**：target 常驻，每步对草稿采样 block 做块对角前向，得到条件于采样前缀的真·on-policy teacher 分布（§2.4）。
- **框架 = verl 0.7.0**（§1 剖析的 Rethink-OPD fork 版本）：三处非标准点（rollout / actor 模型构建与前向 / teacher 打分）均为**子类覆写**而非引擎 fork，且 fork 已现成提供 `token_reward_direct` + 3D top-k policy loss 的 reward→advantage→loss 全链路，可直接搭建（§2.6）。曾评估的 verl 0.8.0 因实现过重转为备选，完整调研见附录 [`DSpark-OPD-verl080.md`](./DSpark-OPD-verl080.md)。

下面逐点细化，并对每一点给出**可行性判断（✅ 高 / ⚠️ 中，有前置条件 / ⛔ 阻塞项）**。代码实现细节留待后续（第 3 部分开发计划及之后）。

---

### 2.1 DSpark 与标准 LLM-OPD 的结构性差异（为什么不能照搬 §1）

| 维度 | 标准 OPD（§1） | DSpark-OPD | 影响 |
|---|---|---|---|
| **Teacher** | 外部**更强**模型，须带来学生「没见过的新能力」 | **Target 模型自身**（Qwen3-4B） | 目标不是能力迁移，而是**逼近 target 分布以最大化接受率**——是「为投机解码服务的自蒸馏」 |
| **Student** | 一个完整的 LLM（独立权重） | **依附于 target 的草稿头**：以 target 中间层 hidden 为 context、block 并行、带 markov/confidence 头 | 学生 forward 结构完全非标准，vLLM/HF-CausalLM 接口不适用 |
| **Rollout** | 学生自回归生成**整条 response** | 在 target-response 的多个 anchor 上做**块并行采样**（每块 `block_size=7` 个 token） | rollout 不是一条连续序列，而是同一条 response 上 512 个「反事实分支」 |
| **词表** | 须共享/可映射（约束） | **天然同词表**（同 tokenizer） | 无对齐问题 ✅ |
| **输入数据** | 只含 prompt + 答案元信息 | 缓存的 target response + target hidden（`CacheDataset`） | 数据管线复用 DeepSpec 现有缓存 |
| **rollout 引擎** | vLLM | **不可用 vLLM**，须 PyTorch 自研（复用 eval 代码） | 见 §2.6 IP-1（自研 rollout worker） |
| **监督目标** | 匹配 teacher 的 next-token 分布 | 同左，但用于**投机解码接受率** | 评估口径不同（接受长度而非任务分数） |

**结论**：Student/Teacher 的「角色」可映射到 verl 的 actor / teacher 抽象，但 **rollout 与 model 两处与 verl 的默认假设强冲突**，构成 §2.6 的三处非标准接入点，是本方案的主要工程量所在。

---

### 2.2 角色映射：Student = Draft，Teacher = Target　✅

- **Student = `Qwen3DSparkModel`（草稿模型）**，即当前训练的对象（`deepspec/modeling/dspark/qwen3/modeling.py`）。OPD 只训练草稿的可训练部分（backbone 各层、`fc`、`markov_head`、`confidence_head`）；`embed_tokens` / `lm_head` 仍从 target 拷贝并冻结（`base_trainer.py:262-274`）。
- **Teacher = Target 模型（Qwen/Qwen3-4B）**。它提供两类信号，需严格区分（这是可行性的关键，见 §2.4）：
  1. **Context K/V**：target 中间层 hidden（`target_layer_ids=[1,9,17,25,33]`）作为草稿注意力的 context——**与草稿采样无关，可缓存**。
  2. **Rollout 打分**：target 对草稿采样 token 的 next-token 概率——**依赖草稿采样，通常不可缓存**。
- **与 §1 的关键差异**：标准 OPD 要求「teacher 比 student 强、且提供新能力」；DSpark 场景恰恰相反——**我们要的就是 teacher ≡ target**，让草稿尽可能复刻 target 在 target-typical 轨迹上的逐 token 分布。§1 论文「思维模式兼容」的条件在这里自动满足（同一个模型）。

---

### 2.3 Rollout：DSpark 的 on-policy 块采样（复用 eval 采样 + 加 R 维多采样 + top-k 候选）

**定义（对齐用户骨架 + RL rollout 语义）**：给定缓存的一条 target response 及其 hidden，随机采 `num_anchors=32` 个 anchor（`sample_anchor_positions`，`common.py:123`；OPD 用 32，远小于 SFT 的 512），在每个 anchor 处让草稿**采样** `block_size` 个 token 作为一条 rollout。**rollout 次数 `n`（如 4）通过在 batch 维 repeat 缓存样本实现**（标准 verl `gen_batch.repeat(rollout.n)`）——每个副本独立采自己的 anchor + block，是 n 份独立 on-policy 数据。

> **rollout 的两层多样性（已锁定，对齐 Rethink-OPD 默认）**：
> - **rollout 次数 `n`（进 batch 维）**：每缓存样本 repeat `n=rollout.n=4` 次，对应 verl「每 prompt 采 n 条」。因用 `token_reward_direct`（逐条独立、**无 GRPO 组内基线**），n 个副本不需共享状态，直接进 batch 维（`B → B×n`）用 verl 标准机制，无需自定义 batch 分块。
> - **不同状态覆盖 `A=32`**：每条 response 上 32 个 anchor 覆盖不同状态（DSpark 特有）。
> - **为何 `n` 进 batch 而非样本内维**：OPD 须常驻 teacher，故 `A` 必小（32），`B×n` 不爆，可直接用 verl；比"n 内嵌样本"更简单（张量无 R 维、复用 verl batch 归一化）。代价：n 个副本各跑一次 backbone（A 小，相对 teacher 全词表 forward 是小头）。
> - **token 级默认 top-k 稠密**（`K=16`, `only_stu`, `student_p`）：每 block 位置取 top-K 候选算加权反向 KL，而非仅采样 token；`K=0` 退化为单采样 OPD。见 §2.5。

**关键实现事实——rollout 复用现有代码**：

- 草稿的 block backbone forward 在训练里**本来就是对所有 block 并行**跑的（`Qwen3DSparkModel.forward` → `_forward_backbone`，`modeling.py:388/361`）；eval 的单块版本是 `forward_dspark_draft_block`（`draft_ops.py:22`）。
- 采样用 `markov_head.sample_block_tokens`（`markov_head.py:55/227`）——块内**自回归**：位置 k 的 markov 偏置条件于**上一个被采样的 token**。这正是 eval 的行为（`build_dspark_proposal` → `sample_draft_tokens`，`draft_ops.py:96`）。

**决定性发现——on-policy 改动是「外科手术式」的，backbone 不变**：

DSpark 的注意力 mask（`create_dspark_attention_mask`，`common.py:86-96`）令每个 draft query 只 attend 到 context `[0, anchor)` 与**同块的 draft 位置**；而 draft 的 KV 输入恒为 `[anchor_token, mask, mask, …]`（`create_noise_embed`，`common.py:264`）。因此：

> **草稿 backbone 的 block hidden 只由 (context, anchor token, mask 嵌入) 决定，与采样出的 token 无关。** 唯一依赖采样 token 的是 **markov 头**（`prev_token_ids`）和 **confidence 头**。

推论：

1. **on-policy 训练的 backbone forward 与现在完全相同**（都喂 mask token），无需改。
2. on-policy 真正带来收益的是 **markov 头**与 **confidence 头**——它们在训练时被 teacher-forcing（真实 prev token）、eval 时却用采样 prev token，存在 **exposure bias**。OPD 恰好在草稿「自己会到达的状态」上训练这两个头，消除该 bias。
3. **rollout 次数 `n` 在 batch 维**：n 个副本各自独立跑 backbone（不共享），但因 A 小(32)、backbone 相对 teacher 全词表 forward 是小头，可接受；换来复用 verl 全套 batch 机制、无自定义分块。

**可行性：✅ 高**。rollout 复用 `deepspec/eval/dspark/draft_ops.py` + `markov_head.sample_block_tokens`；block_rollout 本身 R-agnostic，新增只是 **top-k 候选提取**（`n` 由 verl 在 batch 维 repeat 实现）。

---

### 2.4 Teacher 打分：路线 A（online target 打分）【已锁定】

**决策**：采用 **路线 A——target 模型常驻，每步对草稿的采样 block 做 online 前向打分**，得到条件于**采样前缀**的真·on-policy teacher 分布。这是用户骨架「用 Teacher Model 对 rollout 计算概率」的严格实现。（teacher-forcing 全缓存的近似方案作为降级 fallback 记录在 §2.4.4，非主线。）

要计算 OPD 的 token 级信号 `logπ_target(ỹ_k) − logπ_draft(ỹ_k)`，`logπ_draft` 由草稿自身 logits 得到；`logπ_target(ỹ_k)` 需 target 在草稿采样 token `ỹ_k` 上的概率，且**条件于草稿的采样前缀** `[x_0..x_anchor, ỹ_1..ỹ_{k-1}]`（其中 `x_{≤anchor}` 是真实 token，`ỹ_{<k}` 是草稿采样 token）。

#### 2.4.1 打分机制：target 上的块对角前向（= 批量化的 `verify_draft_tokens`）

对一条 response 的 512 个 block 一次性打分：

1. **context prefill**：target 对真实 response `[x_0..x_{L-1}]` 做一次因果前向，得到全序列的 target KV。此步同时**顺带产出 k=1 位置的精确 teacher 分布**（= 现有缓存的 `aligned_target_logits`，`modeling.py:447-465`），可作正确性对拍。
2. **block 打分**：把 512×`block_size` 个采样 token `ỹ` 作为**追加 query** 拼到 context 之后，position_ids 设为各自 `anchor+1..anchor+block_size`，用**与草稿同构的块对角 mask**（`create_dspark_attention_mask` 的 target 版，`common.py:78`）：每个 block 的 query 只 attend 到自己 anchor 的 context KV `[0, anchor)` + 同块内在先的采样 token。读出这些 query 位置的 target logits → 即 `logπ_target(·)` 条件于采样前缀。
3. 该操作在结构上与 `verify_draft_tokens`（`base_evaluator.py:186`）在 eval 里对**单块**做的 target 验证前向**完全同构**，路线 A 只是把它**批量化到 512 个 block**并去掉接受采样、只取 logits。

**【已锁定】展平批量打分 + 必须先验证等价性**：teacher 打分采用 **A×R×blk 物理展平成一条序列** + 块对角因果 mask + 真实 position_id，一次 target forward 算完全部 block（而非逐 block 循环）。可行的两个前提（见张量契约 S3「维度语义」）：(a) 块对角 mask 隔离 A/R（`q_block_id==kv_block_id`，精准控制每 token attend 对象）；(b) position_id 用各 block 在原序列的真实绝对位置 `anchor+1..anchor+blk`（RoPE 按 position_id 编码，与物理下标无关，故展平不乱）。**⚠️ 块内须因果（下三角），不同于 draft 的块内双向。**

> **S3 实现前必须先做「展平等价性」验证**（张量契约 S3 必做检查）：**展平批量打分** vs **逐 block 独立打分（eval `verify_draft_tokens` 单块）**，最终 `logp_target` 逐 block `allclose`。这是"展平做法可行"的黄金证明——不等价即说明 mask/position_id 有误，展平不可用。

> 与草稿注意力的区别：草稿把 **target hidden 当 context 的 K/V**（DSpark cross-attention）；而路线 A 里 target 用**自己的 KV**（对真实 context 的标准自注意力），追加的采样 block 作为新 query。二者 mask 结构相同、语义不同。

#### 2.4.2 缓存缺口与取舍

- 现有缓存只存了「选定中间层 hidden（`[1,9,17,25,33]`）+ 末层 hidden」，**没有 target 全层 KV**。因此路线 A 的 context KV 无法从缓存重建，需 target 常驻并**每步重新 prefill context**。
- 这意味着路线 A **把 target 前向从离线（缓存一次）搬到在线（每步一次）**，是路线 A 的主要成本来源。
- **缓存分工不变**：草稿仍消费缓存的 `target_hidden_states` 作为其 cross-attention context（与打分无关，照旧缓存、照旧复用 `prepare_target_cache.py` 产物）；路线 A 新增的 target 打分前向是**独立于**该缓存的另一路 target 计算。
- 备选优化（实现期再评估，非阻塞）：缓存 target 全层 KV 以省去每步 context prefill——但 36 层 × 4096 位置 × (K,V) 存储代价高，默认不采用，倾向「target 常驻 + 每步重算 context」。

#### 2.4.3 成本与可行性

- **显存**：target(4B, bf16 ≈ 8GB) 常驻 + 草稿 + 优化器状态。与 eval 同时载 target+draft 的量级一致。
- **计算**：每步每样本一次 target 前向，覆盖 `seq_len + 512×block_size ≈ 4096 + 3584 ≈ 7.7k` 个位置，量级与 eval 反复做 verify 前向相当。
- **可行性：⚠️ 中（可行、偏重）**。无原理阻塞；主要代价是 target 常驻 + 每步 context prefill + block 打分前向，吞吐低于纯缓存训练。这是换取真·on-policy 信号（k≥2 也精确）的必要开销。

#### 2.4.4 Fallback（仅备案，非主线）

若路线 A 的 target 前向成为吞吐瓶颈、需要临时降级：可用缓存 `aligned_target_logits`（条件于**真实**前缀）近似所有 block 位置的 teacher 分布——k=1 精确，k≥2 有「采样前缀 vs 真实前缀」偏差。二者共用同一套 loss 接口，仅「teacher 分布从何而来」不同，可作为 A/B 对照实验的对照组。

---

### 2.5 Loss 映射：从「L1/CE 蒸馏」到「OPD 反向 KL 策略梯度」

**现状回顾**（`deepspec/modeling/dspark/loss.py`，`compute_dspark_loss`）：三项加权，全部 **teacher-forcing**——
- `ce_loss`（α=0.1）：draft_logits 对**真实** target_ids 的 CE；
- `l1_loss`（α=0.9）：draft 概率与**缓存 target 概率**的 L1 距离（`_compute_local_l1_term`）；
- `confidence_loss`（α=1.0）：接受率预测 BCE，目标 `accept_rate = 1 − 0.5·L1`（`_compute_accept_rate_3d`）。

**OPD 版映射**（把「对真实 token 的分布匹配」换成「对草稿采样 token 的反向 KL 策略梯度」）。这里可**直接复用 §1 剖析的 verl 0.7.0 fork 现成机制**（`token_reward_direct` + 3D top-k policy loss，§1.7），只需把 DSpark 的 block 结构对账进其 3D 序列契约（§2.6.4 caveat #2）：

1. **采样候选【默认 top-k 稠密，已锁定】**：在每个 block 位置取草稿 **top-`K` 候选**（`K=16`, `only_stu`），对齐 Rethink-OPD 默认（`LOG_PROB_TOP_K=16`）。`K=0` 退化为单采样 OPD（备选，高方差）。top-k 与 DSpark 现有 `l1_loss`（对分布匹配）思路一致，且论文指出高概率质量集中于极小共享 token 集（97–99%），top-k 足够。
2. **token 级 reward**（§1.7a 的 DSpark 化，对应 fork 的 `compute_distillation_reward`，`dp_actor.py:558`）：
   ```
   rm[block, R, pos, j] = w_j · ( logπ_target_on_topk[j] − student_top_k_logp[j] )   # j ∈ top-K
   ```
   其中 `logπ_target_on_topk` 来自 §2.4 路线 A（online 块对角打分，在学生 top-K 候选上），`w_j = softmax_K(student_top_k_logp)`（`student_p` 加权）。
3. **advantage = `token_reward_direct`**：`A = rm`（§1.7b，`core_algos.py:854`），reward 直接当 advantage，mask 广播。
4. **策略梯度 loss**（§1.7c，`compute_policy_loss_vanilla` 的 3D 分支）：on-policy 下 `L = −Σ eval_mask · A · logπ_draft(候选)`，**对 top-K 维求和**；`n` 个 rollout 副本在 batch 维、loss 直接平均（逐条独立，非 GRPO 组内归一）；block 内位置沿用现有 `loss_decay_gamma` 衰减权重（`_build_loss_weight_mask`，`loss.py:25`）；`eval_mask` 充当 verl 的 `response_mask`。
5. **confidence 头**：目标接受率仍是 `1 − 0.5·L1(draft, target)`，但**在采样 token 上评估**——自然变成「草稿实际会提议的 token 的接受率」，与 eval 口径一致。此项是 DSpark 专有、verl 无对应，须在 fork 的 actor forward / loss 里自行加回。

**与现有三项（`compute_dspark_loss`）的关系**：
- `l1_loss`（teacher-forcing 的对称分布距离）→ 被 OPD 的**采样 token 反向 KL 策略梯度**替代/增强；评估点从真实 token 移到草稿采样 token。
- `ce_loss` 可作为**稳定项保留**（低权重），防止纯 PG 早期方差过大——类似 §1 SFT 冷启动的作用。
- `confidence_loss` 结构不变，仅评估分布改为 on-policy（见第 5 点）。

**可行性：✅ 高（数学）/ ⚠️ 中（落地）**。OPD 的 reward/advantage/PG 数学在 §1 已厘清，且 0.7.0 fork 的 `token_reward_direct` + 3D top-k policy loss 全链路现成；主要落地工作是把 block 结构展平进其 3D 序列契约，以及自行加回 confidence 头（§2.6.4）。

---

### 2.6 框架：verl 0.7.0【已锁定】

**决策**：基于 **verl 0.7.0**（即 §1 剖析的 Rethink-OPD fork 所在版本）实现 DSpark-OPD。

> 决策变更说明：曾评估 **verl 0.8.0**，但经源码实读判定其**实现过重**——0.8.0 是大规模架构重构，DSpark 的三处非标准点（非 vLLM rollout、非 CausalLM 草稿头 actor、块对角 teacher forward）恰好都落在其硬编码的「硬缝」上，均需 fork engine/worker。完整调研见附录 [`DSpark-OPD-verl080.md`](./DSpark-OPD-verl080.md)。**核心反差：0.8.0 移除/重构掉的，恰是 DSpark-OPD 最需要的现成件**（`token_reward_direct`、支持 3D top-k 的 policy loss、torch-forward 的 teacher worker），而这些**在 0.7.0 fork 里都现成可用**。

#### 2.6.1 为什么 0.7.0 更轻（结构性原因）

0.7.0 的 FSDP worker **直接在单进程内持有** actor 模块、rollout 对象、teacher 模块三者，所有触及模型的操作都汇聚到**少数几个可被子类覆写的单一方法**，而非 0.8.0 的 engine/rollout-manager 抽象。因此 DSpark 的三处非标准点在 0.7.0 都是「**子类覆写 4~5 个方法 + 一个 rollout 类**」，无需 fork 引擎。

> ⚠️ 前提澄清：Rethink-OPD fork 自己的 OPD 示例用的是 `rollout.name=vllm` + 标准 HF `reward_model` teacher（`examples/ppo_trainer/on_policy_distillation.sh`）。它新增的 **top-k 蒸馏 reward/loss 机制**与我们的三处非标准点是**正交**的——即 fork **并未**替我们解决「非 vLLM rollout / 非 CausalLM actor」，这两处仍需自建；但 fork 已把 **reward→advantage→loss 的 3D 管线**铺好，可直接搭建其上。

#### 2.6.2 三处非标准点的 0.7.0 接入方式（均为子类覆写，无引擎 fork）

| # | 集成点 | 0.7.0 现状（实读） | DSpark 接入 | 工程量 |
|---|---|---|---|---|
| **IP-1** | Rollout | worker 在**单进程内**直接持有 `self.rollout` 并同步调 `generate_sequences`（`fsdp_workers.py:934`）；rollout 经 `_ROLLOUT_REGISTRY` 按 `(name,mode)` 选类（`rollout/base.py:80`）。⚠️ `HFRollout`/`NaiveRollout` 已**失修/未接线**（`super().__init__()` 签名过时、不在 registry），**不能开箱即用** | 写一个 `BaseRollout` 子类做 512-anchor 块并行采样（复用 `deepspec/eval/dspark/draft_ops.py` + `markov_head.sample_block_tokens`）；加一行 registry 或覆写 `_build_rollout`；因 actor 与 rollout 同进程共享 FSDP 模块，可直接引用 `self.actor_module_fsdp`、`free_cache_engine=False` 跳过 vLLM 权重同步 | 一个类 + 一行注册 |
| **IP-2** | Actor / Model | 模型在 `_build_model_optimizer`（`fsdp_workers.py:271`）经 `AutoModelForCausalLM.from_pretrained` 构建（硬编码），但下游 FSDP/优化器只需一个 `nn.Module`；`_forward_micro_batch`（`dp_actor.py:86`）假设标准 CausalLM，取 `output.logits` 形状 `(B,T,V)` 再按序列 shift-by-1 | 覆写 `_build_model_optimizer` 构建 `Qwen3DSparkModel`（FSDP/优化器代码复用）；覆写 `DataParallelPPOActor._forward_micro_batch` 适配 `(B, num_blocks, block_size, V)` 布局（target hidden 经 DataProto 直接透传进 forward） | 两个方法覆写 |
| **IP-3** | 数据流 | 标准 prompt parquet → tokenizer → DataProto | 把 `CacheDataset`（`prepare_target_cache.py` 产物）桥接为 DataProto，携带 `input_ids/loss_mask/target_hidden_states/target_last_hidden_states` | 自定义 dataset |
| **IP-4** | Teacher 打分 | teacher = `RewardModelWorker`，用**普通 torch forward** `self.reward_module(...)`（`fsdp_workers.py:1993/2229`）——**非 vLLM 绑定**；已实现输出逐 token 3D `(B,T,K)` 张量（`_compute_teacher_top_k_log_probs`、`compute_rm_score`，`fsdp_workers.py:1830/2555`） | 覆写 `RewardModelWorker._build_model` 构建 target；覆写其 `_forward_micro_batch` 用**块对角 mask** 对草稿采样 block 打分（§2.4.1）；`compute_rm_score` 的 3D 打包逻辑复用 | 两个方法覆写 |

#### 2.6.3 verl 0.7.0 fork 现成可用的 OPD 机制（0.8.0 已无）

以下 reward→advantage→loss 全链路在 fork 中**已存在且端到端自洽**（3D 张量从 log-prob → reward → advantage → loss 贯通），可直接搭建：
- **`token_reward_direct` advantage（3D-aware）**：`core_algos.py:854-880`，reward 直接当 advantage，3D 时把 `response_mask` 广播到 `(B,T,1)`；另有 `token_reward_direct_plus_grpo` 变体。
- **支持 3D `(B,T,K)` top-k 的 policy loss**：`compute_policy_loss_vanilla`（`core_algos.py:1058-1197`）的 3D 分支（`1118-1155`）算逐 top-k PPO ratio/clip 后对 K 轴 `sum`（`1144`）。
- **actor 侧蒸馏 reward 组装**：`compute_distillation_reward`（`dp_actor.py:451-624`），含 `student_p/teacher_p/none` 权重与 5 种 union/intersection 策略。
- **`update_policy` 已路由 3D advantage/log-prob**（`dp_actor.py:826-903`），含 on/off-policy old-log-prob 处理。
- **完整 trainer 编排**：`compute_log_prob`（学生 top-k）→ `compute_rm_score`（teacher）→ `compute_distillation_reward`（actor）→ `compute_advantage`（`ray_trainer.py:1107-1144`），由 `use_rm` 门控。

加上 verl 通用基础设施：Ray 编排、FSDP、动态批、checkpoint、日志。

#### 2.6.4 已知 caveat（须在开发计划中预留工作量）

1. **`HFRollout`/`NaiveRollout` 失修**：不能指望开箱 PyTorch rollout，须自写 `BaseRollout` 子类（量小但非零）。
2. **4D block 布局 vs 3D 序列契约的形状对账（最主要工作量）**：fork 的现成 3D 机制里「3D = `(B, response_len, K)`」，而 DSpark rollout 产出 `(B, num_blocks, block_size, V)`。复用 `token_reward_direct` 与 3D `compute_policy_loss_vanilla` 前，需把 **block 展平进序列轴**（512×block_size 映射到 response_len 维），并相应构造 `response_mask`（由 `eval_mask` 充当）。
3. **rmpad/ulysses 关闭**：`_forward_micro_batch` 的 remove-padding / 序列并行机制为标准序列设计；块并行模型宜 `use_remove_padding=False` 并自写更简洁的 forward。
4. **confidence 头无 verl 对应**：DSpark 专有，须在覆写的 actor forward / loss 中自行加回（§2.5 第 5 点）。
5. **checkpoint 兼容**：预留一步 checkpoint 转换，导出为 `Qwen3DSparkModel.from_pretrained` 可读格式，以保 §2.7「eval 零改动」。

**推荐落地形态**：在 Rethink-OPD 的 verl 0.7.0 树内建一个 **DSpark recipe 目录**，通过子类化 `ActorRolloutRefWorker` / `DataParallelPPOActor` / `RewardModelWorker` 的上述 4~5 个方法 + 一个 rollout 类落地，最大化复用已有 3D 蒸馏管线。

> **可行性判断：⚠️ 中（可行，明显轻于 0.8.0）**。三处非标准点在 0.7.0 均为**子类覆写**而非引擎 fork，且 reward→advantage→loss 的 3D 管线现成；主要工作量集中在 caveat #2 的 4D→3D 形状对账。相对 0.8.0（需 fork engine/worker 并重加 token 级蒸馏机制），0.7.0 是**决定性更轻**的承载目标。

#### 2.6.5 Actor 的 FSDP 配置【已定，经源码实测】

DSpark actor 是非标准模型（`Qwen3DSparkModel`，冻结 `embed_tokens`/`lm_head`，rollout 按名调 `_forward_backbone`/`embed_tokens`/`compute_logits` 等**子方法**）。经对 verl 0.7.0 源码实测，确定 actor 的 FSDP 配置如下（与 DeepSpec 训练对齐）：

| 配置 | 取值 | 理由（实测/源码依据） |
|---|---|---|
| **`sharding_strategy`** | **`NO_SHARD`**（直接指定，绕过 `get_sharding_strategy`） | 我们要的是**"数据并行、模型不并行"**：每卡持完整 actor 参数（rollout 子方法能拿到完整参数）+ 梯度跨卡 all-reduce。**这正是 `NO_SHARD`**。⚠️ **不能用 `fsdp_size=1`**：verl `get_sharding_strategy`（`fsdp_workers.py:111-120`）只出 `1D→FULL_SHARD`/`2D→HYBRID_SHARD`、**无 `NO_SHARD` 出口**；`fsdp_size=1` 落进 `(ddp=world,fsdp=1)` 的**退化 HYBRID mesh**——分片维=1 虽让每卡持完整参数，但其**跨卡梯度 all-reduce（replicate 维）在本 PyTorch/FSDP1 版本未被激活 → 多卡梯度不同步、副本静默发散**（`fsdp_workers.py:161` 有 `TODO: support FSDP hybrid shard` 佐证该分支未完善）。**对照实测（2-rank，各喂不同数据，step 后比参数）**：DDP / FSDP `NO_SHARD` / FSDP `FULL_SHARD` 三者 `Δparam=0`（梯度均 all-reduce）；唯 `HYBRID(fsdp_size=1)` `Δparam≈2`（不同步）。故直接 `sharding_strategy=NO_SHARD`（`fsdp_size` 值此时不参与，mesh 用 1D 或省略）。 |
| **`use_orig_params`** | **`True`** | verl 默认 `False`（性能优先：flatten 成单个 `FlatParameter`）。但我们**冻结部分参数**（embed/lm_head `requires_grad=False`）——`False` 下同一 flat 参数内混合 `requires_grad` 会**报错**；`True` 才能逐参数尊重 `requires_grad` 且按原名访问子模块。DeepSpec 训练正是 `use_orig_params=True`（`base_trainer.py:60`）。**对我们是必需，非可选。** |
| **`MixedPrecision`** | `param_dtype=bf16, buffer_dtype=bf16` | 与 `from_pretrained(dtype=bf16)` 及 DeepSpec 训练一致，避免 FSDP 包裹时精度漂移。 |
| **冻结** | `embed_tokens`/`lm_head` `requires_grad=False` | 从 target 拷贝并冻结（§2.2），只训 backbone/`fc`/markov/confidence。 |

> **`NO_SHARD` 的准确语义**：**不做参数分片（模型不并行），每卡持完整 actor 参数 + 梯度跨卡 all-reduce（数据并行）**——同时满足 rollout（子方法调用需完整参数）与 loss（梯度同步）。FSDP 包裹本身仍有非分片副作用（mixed_precision 精度、`use_orig_params` 参数暴露），故仍须由 S2「裸模型 vs FSDP actor logits 对拍」确认数值不变。
>
> **修订说明（S2→S4 迭代）**：曾用 `fsdp_size=1`（误以为其 HYBRID(1) mesh 等价 DDP）。5-agent review + 对照实测发现该退化 mesh **多卡梯度不同步**（root-unit 训练头 `fc`/`markov_head`/`confidence_head`/norms 的梯度永不 all-reduce），已改 `NO_SHARD`。**另一层**问题——训练前向曾绕过 `FSDP.forward`（子方法直接调，连 hook 都不注册）——已在 §S4 修正为经 `module(...)`。两者叠加才是完整修复：`NO_SHARD`（mesh 正确同步）+ 训练走 `FSDP.forward`（hook 注册）。详见 memory [[dspark-opd-multigpu-grad-sync-bug]]。
>
> **备注（NO_SHARD 已弃用）**：PyTorch 标记 `NO_SHARD` deprecated（建议改用 `DistributedDataParallel`）。当前仍可用且保留全套 FSDP 脚手架（checkpoint_manager、`FSDP.clip_grad_norm_`、`summon_full_params`），故先用 `NO_SHARD` 最小改动；若将来失效，回退方案是用 `torch DDP` 包裹 actor（语义相同）。

---

### 2.7 评估：完全复用 `scripts/eval/eval.sh`　✅

- OPD 训练只改变**如何更新草稿权重**，不改变草稿的**结构**（仍是 `Qwen3DSparkModel`，含 block/markov/confidence）。
- 只要 OPD 训练按现有格式产出 checkpoint（`~/checkpoints/<project>/<exp>/step_*`），`eval.py` 的 `EVALUATORS["Qwen3DSparkModel"]` 即可直接加载评测（`eval.py:10-16`）。
- 评测指标（接受率 `accept_rate@k`、平均接受长度 `acceptance_length`、verify_rate）正是 OPD 想优化的量——**评估口径与训练目标一致**，无需任何修改 `eval.sh`。
- **可行性：✅ 高**。零改动。唯一约束：OPD checkpoint 的目录结构与 `from_pretrained` 兼容。

---

### 2.8 可行性结论小结

| 要素 | 锁定方案 | 可行性 | 前置条件 / 风险 |
|---|---|---|---|
| Student/Teacher 映射 | draft ← target 自蒸馏 | ✅ 高 | 同词表、冻结 embed/lm_head，天然成立 |
| Rollout | 复用 eval 块采样（markov 自回归），自研 rollout worker | ✅ 高 | backbone 不变，仅 markov/confidence 头受益；接入 verl 见 IP-1 |
| 输入数据（context） | 复用缓存 target hidden，桥接为 DataProto | ✅ 高 | 与采样无关，合法可缓存；接入 verl 见 IP-3 |
| **Teacher 打分** | **路线 A：online target 块对角打分** | ⚠️ 中 | target 常驻 + 每步 context prefill；无全层 KV 缓存，需每步重算 |
| Loss | OPD 反向 KL 策略梯度 + 保留 CE 稳定项 | ✅ 高（数学）/ ⚠️ 中（落地） | 复用 fork 现成 `token_reward_direct` + 3D top-k PG；confidence 头/block 展平须自拼（§2.6.4 #2/#4） |
| **框架** | **verl 0.7.0**（Rethink-OPD fork） | ⚠️ 中 | 三处非标准点均为**子类覆写**（4~5 方法 + 一个 rollout 类），非引擎 fork；reward→adv→loss 3D 管线现成 |
| 评估 | 复用 `eval.sh` | ✅ 高 | 须加一步 **checkpoint 转换**为 `from_pretrained` 可读格式（§2.6.4 #5），eval 本身零改动 |

**总体判断：方案可行，无原理性阻塞项。** 主要成本集中在两处（均为工程量，非原理障碍）：
1. **路线 A 的 online target 打分**——target 常驻 + 每步 context prefill + 512-block 打分前向，吞吐低于纯缓存训练，是换取 k≥2 精确 on-policy 信号的必要开销。
2. **verl 0.7.0 的接入**——子类覆写 `ActorRolloutRefWorker` / `DataParallelPPOActor` / `RewardModelWorker` 的 4~5 个方法 + 一个 `BaseRollout` 子类，主要工作量是把 block 结构展平进 fork 现成的 3D 序列管线（§2.6.4）。相对已否决的 verl 0.8.0（需 fork engine/worker，附录 `DSpark-OPD-verl080.md`）明显更轻。

### 2.9 单步数据流（DSpark-OPD：路线 A + verl 0.7.0 fork）

> 各阶段输入/输出张量的精确 shape/dtype/语义见**张量契约** [`opd/tensor-contract.md`](./opd/tensor-contract.md)（开发中须维护并据以核对；本图为概览，契约文件为准）。

```
DataProto 一个 batch (来自 CacheDataset — IP-3；B = B_prompt × rollout.n，n 已在 batch 维):
    (input_ids[B,T], target_hidden_states[B,T,L*H], target_last_hidden_states, loss_mask[B,T])
  │
  【S2 Rollout — IP-1，自研 BaseRollout 子类，复用 deepspec/eval/dspark；R-agnostic】
  ├─ sample_anchor_positions            → A=32 anchor + block_keep_mask [B,A]      (common.py:123)
  ├─ 草稿 backbone 并行前向 (mask 输入)   → block_hidden[B,A,blk,d]                  (modeling.py:361)
  ├─ markov 校正后分布采样               → tokens[B,A,blk] + logp_draft[B,A,blk]    (markov_head.py:78)
  └─ 顺手取 top-k 候选 (K=16, only_stu)  → student_top_k_ids/logp[B,A,blk,K]  (topk(corrected_logits))
  │      （base_logits 仅留作 eval 对拍；采样/logp/topk 均基于 markov 校正后分布；无额外前向）
  │
  【S3 Teacher — IP-4，路线 A，RewardModelWorker 覆写，target 常驻】
  ├─ target 对真实 context 因果 prefill  → target KV（顺带产出 k=1 精确分布对拍）     (对照 modeling.py:447)
  └─ A×blk 展平为 query，块对角因果 mask 前向 → logπ_target_on_topk[B,A,blk,K] 条件采样前缀  (批量化 verify_draft_tokens)
  │
  【Reward → Advantage → Loss — 复用 fork 现成 3D 管线，[A,blk] 展平进序列轴，保留 K】
  ├─ rm[·,K] = w_j·(logπ_target_on_topk − student_top_k_logp)   (top-k 稠密, compute_distillation_reward)
  ├─ advantage = rm  (token_reward_direct, 3D-aware)        (§1.7b, core_algos.py:854)
  └─ 3D dual-clip PG (对 K 求和) : L = −Σ eval_mask·decay·A·logπ_draft(候选) (+ α·CE) + confidence BCE → backward
                                                            (§1.7c, core_algos.py:1058-1197)
```

> 说明：`rollout.n` 已在 batch 维（`B=B_prompt×n`），n 个副本各自独立采 anchor、loss 直接平均（无 GRPO 组内归一）。草稿 backbone forward 与现有训练相同（喂 mask 嵌入），on-policy 收益在 markov/confidence 头；`eval_mask` 充当 `response_mask`；复用 fork 3D 管线需把 `[B,A,blk]` 展平进 `[B, A*blk]`（§2.6.4 #2）；confidence BCE 为 DSpark 专有须自行加回（§2.5 第 5 点）。

---

## 第 3 部分：开发计划（逐阶段开发 + 双层测试）

### 3.0 总则与阅读方式

> **【方法学时效说明】** 本节描述的「两层测试（smoke + E2E）」与 **stage-gate（`stop_after` gate 打印真实中间输出并 `exit(0)`、同一启动命令只前移 gate）** 方法学，是 **S0–S5 逐阶段开发期**的脚手架，用于在每个阶段单独验证一个集成点。**S6 性能优化后已随 gate 机制一并移除**（rollout→teacher→update 融合为单次 `actor_rollout_wg.train_step`，`stop_after`/`STOP_AFTER`/`gate.py` 均删除，见文首「架构演进：融合 + 去 gate」）。下文对 gate、`STOP_AFTER` 的描述与命令均为**历史开发记录**；当前训练/E2E 统一用 `bash run.sh`（NGPUS/BATCH/STEPS/SAVE_FREQ/EXP 旋钮，无 `STOP_AFTER`）。

- **逐阶段推进**：本计划分 8 个阶段（S0–S7）。**每个阶段独立开发、独立验证、通过后再进下一阶段**，不一次性写完全部代码。
- **verl 从头接入、集成风险前置**：从 S0 起即搭真实 verl recipe 骨架，四个集成点先用占位桩，沿 dataflow 逐个变真（§3.0.2）。每个模块从落地起就在真实 verl 调用链里被检验；模块的孤立正确性另由 smoke-test 独立覆盖。

#### 3.0.1 两层测试（分工明确）

每阶段有**两类测试**，职责与执行者不同：

| | **smoke-test** | **end-to-end (E2E) test** |
|---|---|---|
| 执行者 | **我（自动跑）** | **你（人工执行、人工检查）** |
| 粒度 | 模块级、孤立 | 训练 dataflow 级、贯通已实现部分 |
| 判定 | **程序化断言**（`allclose`/`isfinite`/确定性/shape），pass/fail 明确 | **人工看打印的 output** 是否合理 |
| 启动方式 | 直接调被测函数的最小脚本 | **verl 入口 `run.sh`**（同一配置、同一 worker 装配、同一 dataflow）+ `trainer.stop_after=<phase>` |
| 目的 | 证明「这个模块**算得对**」 | 证明「这个模块**在真实 verl 训练链路里、以真实调用方式，产出合理中间结果**」 |
| 中间阶段行为 | — | 未实现的 IP 用占位桩顶着；已实现模块打印其真实 output；到本阶段前沿 gate 处 `exit(0)` |

关键区别：smoke-test 用「捏造的小输入 + 断言」孤立校验；**E2E test 从 S0 起就走真实 verl 启动路径**——未实现模块用占位桩满足契约让 dataflow 能一路跑到当前前沿，已实现模块打印真实中间张量供你核验，然后 `exit(0)`。这样每个模块**一落地就在最终 verl 调用环境里被检验**，集成风险前置到最早（§3.0.2）。

#### 3.0.2 verl 从头接入 + 占位桩 + stage-gate

**核心方法（贯穿全程）**：从 S0 起就搭好**真实的 verl 0.7.0 recipe**（`recipe/dspark_opd/`：`main.py` + `config.yaml` + `run.sh` + 三个 worker 子类 + rollout + dataset）。**四个集成点（IP-1~4）一开始全部用「占位桩」**——桩满足 verl 的调用契约（返回形状正确的占位张量），使整条 dataflow 能被 verl 真实驱动、启动不报错。然后**沿 §2.9 的 dataflow 顺序逐个把桩替换为真实实现**，每替换一个就通过 **verl 入口启动**，在该模块的**下一个前沿**用 `stop_after` 打印刚实现模块的真实 output 并 `exit(0)` 提前退出。

- **占位桩（stub）**：未实现的 IP 用最小桩满足契约。例：rollout 桩返回随机 token + 全 0 logp（形状对）；teacher 桩返回全 0 logp（形状对）；actor 模型桩可先用真实 `Qwen3DSparkModel`（构建不贵）或按需延后。桩的作用是**让 verl 能一路调下去**，从而每个真实模块落地时立刻处在真实调用环境。
- **stage-gate（`trainer.stop_after=<phase>`）**：recipe 的编排读取该配置，在指定 phase（`data` / `rollout` / `teacher` / `loss` / `off`）**完成并 dump 输出后 `exit(0)`**。这就是你要的「`exit(0)` 提前退出」机制——同一 verl 启动命令，靠一个开关控制走到哪一步。
- **演进连续性**：S0 gate 在最前（验证 verl 能启动到 dataflow 入口）；S1 gate=`data`、S2 gate=`rollout`、S3 gate=`teacher`、S4 gate=`loss`（含一次 backward）；S5 起 gate=`off` 放开完整 step。**启动命令、config、worker 装配自始至终同一套**，只是 gate 前移、桩变真。

> 好处正是你指出的：集成风险前置到 S0（verl 装配、Ray、FSDP 包自定义模型、DataProto 契约一次性打通），后续每个模块都在**最终训练的真实调用链**里被检验，杜绝「孤立单测过、集成才炸」。

#### 3.0.3 每阶段测试呈现约定

每个阶段给出两类测试：
- **smoke-test（我执行，自动断言）**：直接 import 该阶段新实现的**纯函数**做孤立正确性校验（`allclose`/`isfinite`/确定性/shape/可逆性），pass/fail 程序化判定。不依赖 verl 启动，是我的快速正确性门。
- **E2E test（你执行，人工检查）**：用 **verl 入口 `run.sh`** 启动，`stop_after` 设到本阶段前沿，你**人工查看打印的真实 output** 是否合理，并确认在 gate 处 `exit(0)`（rc=0）。这是「以最终端到端训练同等方式启动」的检验。

二者互补：smoke 保「算得对」，E2E 保「在真实 verl 链路里、以真实启动姿势、产出合理中间结果」。

**统一的 E2E 启动命令**：模型路径、数据、FSDP/rollout 配置全部固化在 `config/dspark_trainer.yaml`，命令行**不再传** `model.path` 等。

> *（历史：S0–S5 开发期各阶段靠 `STOP_AFTER=<data|rollout|teacher|loss|off>` 前移 gate；现融合去 gate，统一命令如下）*

```bash
bash third_party/verl/recipe/dspark_opd/run.sh
```

（可用 `NGPUS`/`BATCH`/`STEPS`/`SAVE_FREQ`/`EXP` 覆盖卡数、batch、步数、落盘频率、实验名。config 里 `actor_rollout_ref.model.path`=draft `step_latest`、`reward_model.model.path`=target Qwen3-4B、`override_config.dspark_tokenizer_path`=target。）下文各阶段 E2E 历史命令均基于此，仅当时的 `STOP_AFTER` gate 不同。

**张量契约**：各阶段输入/输出张量的精确 shape/dtype/语义定义在 [`opd/tensor-contract.md`](./opd/tensor-contract.md)。每实现/修改一个阶段，先对照契约核对张量，smoke 断言引用契约数值；张量设计变化时**先改契约、再改代码**。

#### 已确认的环境事实（写入前已核实）

| 项 | 事实 |
|---|---|
| 机器 | 8 × NVIDIA H100 80GB |
| 训练专用环境 | **`~/.venv/dspark-opd`**（uv 管理，Python 3.11.6，**当前干净、零包**）——本计划所有命令均用 `~/.venv/dspark-opd/bin/python` |
| DeepSpec 现有运行环境 | `/opt/pytorch/bin/python3`（torch 2.12，**无 transformers**）——仅作参照，不用于 OPD |
| verl 0.7.0 源码 | `/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl`（`version=0.7.0.dev`） |
| DeepSpec requirements | `torch==2.9.1, transformers==5.10.2, numpy==2.4.4, prettytable, tensorboard, triton==3.5.1, ...`（`requirements.txt`） |
| 已有 target cache | `/mnt/scratch/qwen3_4b_target_cache`（manifest：`num_samples=1339825`，`hidden_size=2560`，`hidden_dtype=float8_e4m3fn`，`max_length=4096`，layers `[1,9,17,25,33]`，`block_size=7`） |
| 已有 draft checkpoint | `/mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest`（可作 OPD 初始化 / eval 基线） |
| 目标模型 | `Qwen/Qwen3-4B` |

> ⚠️ **环境依赖冲突是首要风险**：DeepSpec 要 `torch==2.9.1 + transformers==5.10.2`，verl 0.7.0 要 `ray + vllm + flash-attn` 等，且 vLLM 常对 torch/transformers 版本有强约束。S0 的核心任务就是让二者在 `dspark-opd` 一个环境里共存（我们**不使用 vLLM rollout**，这一点大幅放松了对 vLLM 的版本约束——见 S0）。

#### 阶段全景

所有阶段的 E2E 入口都是**同一个 verl `run.sh`**，仅靠 `trainer.stop_after` 前移、占位桩变真（§3.0.2）。

| 阶段 | 名称 | 本阶段把哪个桩变真 | E2E `stop_after` 前沿 | 对应 §2 |
|---|---|---|---|---|
| **S0** | 环境构建 + verl recipe 骨架（4 桩全占位） | —（全部占位） | `data`（走到数据步即停） | §2.6 |
| **S1** | 数据桥接：CacheDataset → DataProto | IP-3 dataset | `data` | IP-3 |
| **S2** | **Actor 构建（`Qwen3DSparkModel` FSDP）+ 块并行 rollout（tokens/logp）** | **IP-2 actor 构建 + IP-1 rollout** | `rollout` | IP-1/IP-2, §2.3 |
| **S3** | Teacher 打分（路线 A 块对角前向） | IP-4 teacher worker | `teacher` | IP-4, §2.4 |
| **S4** | Reward→Adv→Loss（4D block → 3D 序列 + backward） | actor 训练前向 + loss 拼接 | `loss` | §2.5, §2.6.4#2 |
| **S5** | 放开完整单 step（去桩、全链路真实） | —（gate=off，单 step） | `off`（1 step） | §2.6.2 |
| **S6** | 端到端小规模训练（多卡、多 step） | — | `off`（N step） | 全 |
| **S7** | checkpoint 转换 + `eval.sh` 对照基线 | —（`eval.sh` 零改动） | eval | §2.7, §2.6.4#5 |

> **【时效脚注】** 上表「E2E `stop_after` 前沿」列（`data`/`rollout`/`teacher`/`loss`/`off`）是 **S0–S5 逐阶段开发期**的 gate 记录。**训练现已改用融合的 `actor_rollout_wg.train_step`、stage-gate 机制已移除**（见文首「架构演进：融合 + 去 gate」），该列仅作历史阶段划分参考，不再对应可用的启动开关。
> S0 搭 verl recipe 骨架 + 4 占位桩、启动到 `data` gate；此后每阶段替换桩、gate 前移。
> **IP-2（actor 模型构建）并入 S2**：verl 语义下 **rollout 就是"用 actor model 的当前权重采样"**，且 verl worker 强制先 build actor（`fsdp_workers.py:781`）再 build rollout（`:814`）——二者本就一体，原先把 IP-2 留到 S4 是错误切分。故 S2 覆写 `_build_model_optimizer` 构建 `Qwen3DSparkModel` FSDP actor，rollout 直接用 `self.actor_module_fsdp` 采样。S4 复用这个已建好的 actor 做训练前向 + loss。

---

### S0 — 环境构建 + verl recipe 骨架（4 桩全占位）

**目标**：(a) 在干净的 `~/.venv/dspark-opd` 里装齐「DeepSpec 依赖 + verl 0.7.0 运行时子集」，两侧包都能 import；(b) 搭起**真实 verl recipe 骨架** `recipe/dspark_opd/`，四个集成点全用**占位桩**，能通过 `run.sh` 启动 Ray/worker 并走到 `data` gate 后 `exit(0)`。这一步把最大集成风险（环境冲突、verl 装配、Ray、gate 机制）一次性打通。

**交付物**
- ✅ `scripts/opd/setup_env.sh`（安装脚本，你执行）、`scripts/opd/check_env.py`（自检）、`scripts/opd/verl_compat.py`（transformers 兼容 shim）、`docs/opd/env-setup.md`、`docs/opd/pip-freeze.txt` —— **环境部分已完成**。
- `recipe/dspark_opd/`（在 vendored verl 树内：`DeepSpec/third_party/verl/recipe/dspark_opd/`）：
  - `main.py`（verl 入口，**顶部先调 `verl_compat.apply()`**）、`config.yaml`、`run.sh`；
  - `worker.py`：`DSparkActorRolloutRefWorker` / `DSparkRewardModelWorker`（**桩**：构建可延后，`_forward_micro_batch` 先返回形状正确的占位）；
  - `rollout.py`：`DSparkRollout(BaseRollout)`（**桩**：返回随机 token + 全 0 logp，形状对）；
  - `dataset.py`：`DSparkCacheDataset`（**桩**：先返回固定假样本，形状对）；
  - 编排里实现 `trainer.stop_after` gate（`data/rollout/teacher/loss/off`）。

**关键决策 / 做法**
- 安装策略见 `docs/opd/env-setup.md`：numpy 走 1.26（verl 要 `<2.0`，与 DeepSpec 的 2.4.4 冲突，裁决用 verl 的）、transformers 5.10.2、verl `-e --no-deps` 装 + 手工列运行时子集、**不装 vLLM/flash-attn**。
- **transformers 兼容 shim（已落地）**：verl 0.7.0 用的 `AutoModelForVision2Seq` 在 5.10.2 被更名为 `AutoModelForImageTextToText`。`verl_compat.apply()` 在 import verl 前补别名（不改 verl 源码、不降 transformers）。经扫描确认 verl 全库仅此 1 个符号缺失。
- gate 通过在 recipe 编排（fit 循环内）按 `stop_after` 值 dump + `sys.exit(0)` 实现。

**smoke-test（我执行，自动断言）** ✅ 已通过
- `scripts/opd/check_env.py`（import verl 前先 `verl_compat.apply()`）断言：`torch==2.9.1`、`transformers==5.10.2`、`numpy==1.26.x`、`cuda.device_count()==8`；DeepSpec 关键 import 成功；verl `get_adv_estimator_fn("token_reward_direct")` 非空、`ActorRolloutRefWorker`/`RewardModelWorker` 存在。**实测末行 `[S0] ENV OK`，exit 0**。

**E2E test（你执行，人工检查）** ✅ 已通过 — 统一命令（§3.0.3），gate=`data`：

> *（历史：原为 gated `STOP_AFTER=data`；现融合去 gate，命令如下）*

```bash
bash third_party/verl/recipe/dspark_opd/run.sh
```

**你应看到**（实测已确认）：Ray 起 `DSparkTaskRunner`；stub 数据集打印 `[DSparkCacheDataset:S0-STUB] ...`；gate 打印 `[gate] reached phase=data` + 一个 batch 的 6 键摘要（`input_ids (2,32) int64`、`target_hidden_states (2,32,12800) bf16 finite=True` 等）+ `[gate] phase=data, stopping cleanly (rc=0)` + `gated stop ... returning cleanly`；**驱动进程 rc=0**。此步证明 **verl recipe 骨架 + 占位桩 + gate + 数据→DataProto 整链能真实跑起来**——集成地基已成。

**通过标准**：E2E rc=0 且打印到 `data` gate。✅ **S0 全部完成（环境 + 骨架均已实测通过）**。
- **交付物**（recipe 侧，均在 vendored `DeepSpec/third_party/verl/recipe/dspark_opd/`）：`__init__.py`（compat shim + rollout 注册）、`task_runner.py`（`DSparkTaskRunner`）、`main.py`（瘦入口）、`trainer.py`（`DSparkTrainer` gated fit）、`gate.py`（`GateStop`/`maybe_gate`）、`dataset.py`（stub）、`rollout.py`（stub）、`worker.py`（stub 子类）、`config/dspark_trainer.yaml`、`run.sh`。verl 作为第三方依赖 vendored 进 DeepSpec，Rethink-OPD 保持只读（见 `docs/opd/env-setup.md`「vendoring 与 git」）。
- **本阶段实测踩坑并修复**（见 `docs/opd/env-setup.md`）：① Ray worker 侧 shim 未生效 → 把 `DSparkTaskRunner` 移到真实子模块 + shim 内联进包 `__init__`；② Ray actor 里 `sys.exit(0)` 触发 `ActorDiedError`(rc=1) → gate 改为 `raise GateStop` + `run()` 捕获后正常 return。
- **已排除的风险**：numpy/transformers 冲突、`AutoModelForVision2Seq` 兼容、vLLM/flash-attn 硬依赖、Ray/hydra/gate 装配 —— 全部打通。

---

### S1 — 数据桥接：CacheDataset → verl（把 IP-3 桩变真）✅ 已完成

**目标**：把 dataset 桩替换为真实 `DSparkCacheDataset`——读 `/mnt/scratch/qwen3_4b_target_cache`，产出 `input_ids / loss_mask / target_hidden_states / target_last_hidden_states`（后两者 float8_e4m3fn 按 manifest 反量化+sanitize 为 bf16）。rollout/teacher/loss 仍为占位桩。

**实现要点（实测确定）**
- **直接复用 DeepSpec `CacheDataset`**（`deepspec/data/target_cache_dataset.py`）：其 `__getitem__` 已内部完成 fp8 反量化+sanitize，返回 `{input_ids(int32), loss_mask(uint8), target_hidden_states, target_last_hidden_states}`。`DSparkCacheDataset` 只做适配：`input_ids→long`、补 `attention_mask`(ones over 真实长度) 与 `position_ids`(arange，供 verl gen 路径 `_get_gen_batch` pop 用)。
- **变长批处理**：cache 样本是**变长**(T≤4096)，而 verl 默认 `collate_fn` 用 `torch.stack`（要求等长）。故新增 `dspark_collate_fn` **右填充**到 batch 内 T_max，并在 `task_runner.run()` 里替换 verl 默认 collate。`cache_path` 由 `data.dspark.target_cache_path` 提供。

> **变长/padding 三方对比与安全性（已核实）**：
> - **DeepSpec 原生训练**（`train.sh`，默认 `local_batch_size=1` 几乎无跨样本 padding）：`CacheCollator` 亦**右填充+零填充**（与本阶段 `dspark_collate_fn` 同）；padding 屏蔽靠 **`loss_mask`**（不传 attention_mask）——anchor 只在 `loss_mask>0` 采样、`eval_mask` 乘 `loss_mask`、块对角 mask 只 attend context `[0,anchor)` 而 padding 在 anchor 之后，**三重免疫**。
> - **Rethink-OPD**（verl RL）：走 verl 标准「左 padding prompt + 右 padding response + 显式 attention_mask」，且 **开 `use_remove_padding=True`**（rmpad 物理移除 padding），loss 用 `response_mask`——**另一套机制**。
> - **本方案**：填充方式同 DeepSpec（右填充），padding 屏蔽沿用 DeepSpec 的 **loss_mask 路线**，故最终 loss（S4）安全。S1 补的 `attention_mask`/`position_ids` 仅供 verl gen 路径 pop 用；我们的块并行 rollout（S2）会像 DeepSpec 那样**自造块对角 mask + 从 loss_mask 采 anchor**，不依赖该 attention_mask。
> - ⚠️ **S2/S4 必须坐实 `use_remove_padding=False`**（§2.6.4 caveat #3）：rmpad 是为标准序列设计的，与块并行结构冲突；关闭后 padding 屏蔽完全由 loss_mask/eval_mask 承担，与 DeepSpec 训练一致。

**交付物**
- `recipe/dspark_opd/dataset.py`：`DSparkCacheDataset` + `dspark_collate_fn`。
- `task_runner.py`：注入 `dspark_collate_fn`。`config/dspark_trainer.yaml`：`data.dspark.target_cache_path` + `n_samples`。
- `scripts/opd/s1_smoke.py`（孤立正确性校验，我执行）。

**smoke-test（我执行，自动断言）** ✅ 已通过

```bash
PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec \
~/.venv/dspark-opd/bin/python scripts/opd/s1_smoke.py --cache /mnt/scratch/qwen3_4b_target_cache --n 4
```
断言（实测全 OK）：len==`min(n, 1339825)`；6 键齐全；`input_ids` long、hidden `[T, 5*2560=12800]`/`[T, 2560]`、bf16、`isfinite`；`dspark_collate_fn` 右填充到 `(B, T_max, ...)` 且 `attention_mask` 行和==各样本真实长度。实测样本 T=226、`input_ids[:8]=[151644, 8948, 198, ...]`（真实 Qwen token）。

**E2E test（你执行，人工检查）** ✅ 已通过 — 统一命令（§3.0.3），gate=`data`，数据步已是真实缓存：

> *（历史：原为 gated `STOP_AFTER=data`；现融合去 gate，命令如下）*

```bash
bash third_party/verl/recipe/dspark_opd/run.sh
```

**你应看到**（实测已确认）：`[DSparkCacheDataset:S1] cache=... len=1339825 using=8 ...`；gate=`data` 打印**真实缓存 batch**——`input_ids (2,244) int64 head=[151644, 8948, 198, ...]`、`target_hidden_states (2,244,12800) bf16 finite=True head=[40.0, -8.0, 1.75, ...]`（**非全 0**，对照 S0 假样本）；rc=0。

**通过标准**：smoke 全断言通过；E2E 打印真实样本、rc=0。✅ **S1 完成**。

---

### S2 — Actor model 构建 + 用 actor 做块并行 rollout（IP-2 + IP-1）✅ 已完成

> **状态**：actor FSDP 构建 + 块并行 rollout（`tokens`/`logp_draft [B,A,blk]` + top-K 候选 `student_top_k_ids/logp [B,A,blk,K]`）+ GPU 采样 + eval 一致性 smoke **已通过**（gate=`rollout` rc=0，`num_anchors=32`, `K=16`）。rollout 次数 `n` 由 verl `rollout.n` 在 batch 维实现；top-k 候选在 rollout 顺手 `topk(corrected_logits)` 取（DSpark 省了 Rethink-OPD 的独立学生前向，见张量契约「前向次数账」）。S2 输出与设计契约一致。

**目标**：两件一体的事——(a) **构建 actor model**：覆写 `_build_model_optimizer`（`fsdp_workers.py:271`）用 `Qwen3DSparkModel`（从 `step_latest` 加载）替代 `AutoModelForCausalLM`，FSDP 包裹为 `self.actor_module_fsdp`；(b) **用 actor 做 rollout**：`DSparkRollout` 用 actor 权重块并行采样，产出 `tokens [B, A, blk]` + `logp_draft` + top-K 候选 `student_top_k_ids/logp`（`topk(corrected_logits, K)`，见张量契约 S2）。teacher/loss 仍为桩。

> **为何 IP-2 并入 S2（概念澄清）**：verl 里 **actor model = 正在被训练的策略权重**（`self.actor_module_fsdp`），同时扮演 train / rollout / ref 三角色（`fsdp_workers.py:189-191`）；**rollout 的定义就是"用 actor 的当前权重生成"**（on-policy 的根本要求：每步用学生此刻的权重采样）。因此 rollout 与 actor 构建**本就一体**，且 verl 强制先 build actor（`:781`）再 build rollout（`:814`）。我们的 `Qwen3DSparkModel` 非标准 CausalLM（`architectures=['Qwen3DSparkModel']`，不在 HF Auto 映射），故必须覆写 `_build_model_optimizer`；构建后 rollout 从 `self.actor_module_fsdp` 取权重（同进程共享，无需 vLLM 权重 sync）。
>
> **smoke 用的就是 actor model**：smoke 里 `Qwen3DSparkModel.from_pretrained(step_latest)` 加载的**正是 actor model 在初始（step 0）状态**——未经训练更新时，actor 权重 ≡ checkpoint 权重。与 verl 路径的唯一差别是 FSDP 包裹（单卡 forward 数值一致）。故 smoke 验证的就是真实 actor 的采样正确性，非"另一个静态模型"。

> **⚠️ 参考实现的重要澄清（已核实 `scripts/eval/eval.sh`）**：eval 推理是**严格 `bsz=1`**（`base_evaluator.py:331` 硬断言 `assert input_ids.size(0)==1`），逐样本循环、**无 batch、无 padding**；且走的是**增量式**路径（`forward_dspark_draft_block` 带 `past_key_values_draft` DynamicCache、逐块推进），与训练的**块并行**路径是两套 forward。因此：
> - S2 应复用的是**训练那套块并行 forward**（`Qwen3DSparkModel.forward` 的 `_forward_backbone` + `create_dspark_attention_mask`，本就支持 bsz>1），**而非** eval 的单块增量路径；eval 代码只作「块内 markov 采样」与「单块正确性对拍」的参考。
> - **右填充对块并行 rollout 是安全的**（屏蔽机制与 DeepSpec 训练同源，靠 loss_mask 不靠 attention_mask）：anchor 只在 `loss_mask>0` 采（padding 处=0，永不被选）；块对角 mask 用 `anchor_positions[b]`/`block_keep_mask[b]` 逐样本索引（无跨样本泄漏）；draft 块只 attend context `[0,anchor)`，padding 在 anchor 之后永不被 attend。
> - **两个 eval 从未验证、S2 必须显式处理的点**：
>   1. **批不变性**：eval 是 bsz=1，批量化是全新路径。须验证「样本单独跑」与「它在 padded batch 里跑」得到**相同 block logits**（S2 关键 smoke 断言）。
>   2. **无效块 NaN**：`block_keep_mask=False` 的块在块对角 mask 下整行被 mask（softmax 全 -inf → 可能 NaN）。训练靠 `eval_mask` 把它们排除出 loss，但 rollout 若对这些块采样会踩 NaN → **采样前必须按 `block_keep_mask` 排除无效块**。

> **⚠️ 参考实现的重要澄清（已核实 `scripts/eval/eval.sh`）**：eval 推理是**严格 `bsz=1`**（`base_evaluator.py:331` 硬断言 `assert input_ids.size(0)==1`），逐样本循环、**无 batch、无 padding**；且走的是**增量式**路径（`forward_dspark_draft_block` 带 `past_key_values_draft` DynamicCache、逐块推进），与训练的**块并行**路径是两套 forward。因此：
> - S2 应复用的是**训练那套块并行 forward**（`Qwen3DSparkModel.forward` 的 `_forward_backbone` + `create_dspark_attention_mask`，本就支持 bsz>1），**而非** eval 的单块增量路径；eval 代码只作「块内 markov 采样」与「单块正确性对拍」的参考。
> - **右填充对块并行 rollout 是安全的**（屏蔽机制与 DeepSpec 训练同源，靠 loss_mask 不靠 attention_mask）：anchor 只在 `loss_mask>0` 采（padding 处=0，永不被选）；块对角 mask 用 `anchor_positions[b]`/`block_keep_mask[b]` 逐样本索引（无跨样本泄漏）；draft 块只 attend context `[0,anchor)`，padding 在 anchor 之后永不被 attend。
> - **两个 eval 从未验证、S2 必须显式处理的点**：
>   1. **批不变性**：eval 是 bsz=1，批量化是全新路径。须验证「样本单独跑」与「它在 padded batch 里跑」得到**相同 block logits**（S2 关键 smoke 断言）。
>   2. **无效块 NaN**：`block_keep_mask=False` 的块在块对角 mask 下整行被 mask（softmax 全 -inf → 可能 NaN）。训练靠 `eval_mask` 把它们排除出 loss，但 rollout 若对这些块采样会踩 NaN → **采样前必须按 `block_keep_mask` 排除无效块**。

**交付物**（均 ✅ 已实现并通过 smoke + E2E）
- `recipe/dspark_opd/block_rollout.py`：`dspark_block_rollout(model, ...)` —— 块并行 rollout 内核（复用训练 forward + markov 采样；显式排除无效块）。
- `recipe/dspark_opd/worker.py`：
  - `DSparkActorRolloutRefWorker._build_dspark_module`（构建 `Qwen3DSparkModel`，bf16 + flex_attention，冻结 embed/lm_head）+ `_build_model_optimizer`（IP-2：FSDP 包裹 `ShardingStrategy.NO_SHARD`（config `fsdp_size=-1`；**非** `fsdp_size=1`，见 §2.6.5）/`use_orig_params=True`/MixedPrecision bf16 + 优化器；设 `generation_config=None`）+ `_build_rollout`（`set_module(actor)` attach 给 rollout）；
  - `DSparkRewardModelWorker._build_model`（把 verl 硬编码的 `flash_attention_2` patch 为 `sdpa`，S2 仅为让 teacher worker init 成功）+ **`compute_rm_score`（S3 覆写，调 `score_blocks_flat` 块对角打分）**。
- `recipe/dspark_opd/rollout.py`：`DSparkRollout.set_module` + `generate_sequences` 用 attach 的 actor 权重调 `dspark_block_rollout`（bf16）。
- `recipe/dspark_opd/config/dspark_trainer.yaml`：actor 补 `ppo_micro_batch_size_per_gpu` + `fsdp_config(fsdp_size=-1/use_orig_params/mixed_precision bf16)`（worker 内直接 `NO_SHARD`，见 §2.6.5）；rollout 补 `tensor_model_parallel_size=1`。
- `recipe/dspark_opd/trainer.py`：gated fit 的 `rollout` phase（调 `actor_rollout_wg.generate_sequences(batch)` → dump → gate）。
- `scripts/opd/s2_smoke.py`：孤立校验，含 eval 一致性测试。

> **★ 与 eval 的一致性测试（本阶段最强正确性校验）**：验证我们的批量块并行 rollout 与 `scripts/eval/eval.sh` 的推理内核在**同样样本、同样 anchor**下 logits 一致。
> - **注意**：不能跑完整 eval loop——`generate_decoding_sample` 是自回归、自选 anchor、DynamicCache 增量 context，无法「指定 anchor」。故复用 eval 的**计算内核**喂受控输入：
>   - 我们侧：批量块并行 forward（右填充 batch + `BlockMask`，一次算所有 anchor）→ 取选定 anchor 的 `base_logits_ours`；
>   - eval 侧：对每个选定 anchor `a`，用 eval 的 `forward_dspark_draft_block`（context = 该 anchor 之前的真实上下文 + 单块，DynamicCache 路径）→ `build_dspark_proposal` 得 `base_logits_eval`；
>   - 断言 `allclose(base_logits_ours[a], base_logits_eval)` 逐 anchor（核差异：我们用 flex_attention BlockMask，eval 用单块增量路径，由容差兜住——正是要验证的等价性）。
> - 若含 markov 头/confidence 头，同样对拍其 block logits / confidence logits。

**smoke-test（我执行，自动断言）** ✅ 已通过（实测见下）

```bash
PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec:/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl \
~/.venv/dspark-opd/bin/python scripts/opd/s2_smoke.py \
    --draft /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest \
    --cache /mnt/scratch/qwen3_4b_target_cache --num-anchors 64 --temperature 1.0 --seed 42
```

**实测结果**（2 条真实缓存样本 real_len=[226,244]，padded T=244，各 64 有效块；`tokens/logp_draft` shape `(2,64,7)`）：

| 检查 | 判据 | 实测 | 结论 |
|---|---|---|---|
| **★ eval 一致性** | 相对最大偏移 `relmax<2e-2` 且贪心 token 一致 | 5 个 anchor：`max|Δ|`=0.14~0.19、`|logit|max`≈23~26、**`relmax`=6.1e-3~7.4e-3**、`argmax_agree`=True | ✅ 批量 flex 路径与 eval 单块 sdpa 内核数值等价（差异仅 bf16 flex-vs-sdpa kernel 噪声，最大相对偏移 0.7%，不改贪心 token） |
| **批不变性** | 相对均值误差 `<5e-3` 且 argmax 100% | `max|Δ|`=0.25、`mean|Δ|`=5.0e-3、logit 尺度 3.09、**相对均值误差 1.6e-3**、argmax 100% | ✅ 右填充不泄漏；仅 bf16/flex 分块 tiling 噪声（padding 若泄漏 mean|Δ| 会大得多） |
| **无 NaN** | 有效块 `logp_draft` 有限且 ≤0 | 全通过 | ✅ 无效块已排除 |
| **anchor 合法性** | 有效块 anchor ∈ `loss_mask>0` | 全通过 | ✅ padding 处不采 anchor |
| **确定性** | 同 seed 两次 `tokens` 一致；temp=0 两次一致 | 全通过 | ✅ |
| **top-K 候选** | shape `[B,A,blk,K]`、logp 降序有限、取自 corrected 分布 | `top_k_ids (2,64,7,16)`、降序、top-1==采样 token | ✅ 候选与采样同源（markov 校正后） |

> **`max|Δ|` / `relmax` 定义**：`max|Δ|` = 一块 logits（`blk×V`=7×151936）逐元素 `|ours−eval|` 的最大值；`relmax` = `max|Δ| / |logit|max`（相对该块 logits 绝对值尺度）。绝对偏移 0.19 看似不小，但相对信号（尺度~25）仅 0.7%，且不改变任何被采样/贪心 token。

**E2E test（你执行，人工检查）** ✅ 已通过 — 统一命令（§3.0.3），gate=`rollout`（构建 **FSDP DSpark actor** + 用它块并行采样）：

> *（历史：原为 gated `STOP_AFTER=rollout`；现融合去 gate，命令如下）*

```bash
bash third_party/verl/recipe/dspark_opd/run.sh
```

**实测结果**（rc=0，`num_anchors=32`, `K=16`）：构建 `Qwen3DSparkModel` FSDP actor → attach 给 `DSparkRollout` → 用 actor 权重块并行采样 + top-k，gate=`rollout` 打印真实输出：`rollout_tokens (2,32,7)` int64 finite、`rollout_logp_draft (2,32,7)` finite ≤0、`rollout_student_top_k_ids/logp (2,32,7,16)`（top-1 候选 == 采样 token、logp 降序）、`block_keep_mask/eval_mask` 齐全；末行 `[gate] phase=rollout, stopping cleanly (rc=0)`。

**E2E 一路踩通并修复的 6 个真实集成问题**（均由 E2E 暴露，体现"从头接入 verl"价值）：
1. teacher(reward) `_build_model` 硬编码 `flash_attention_2`（未装）→ `DSparkRewardModelWorker._build_model` patch 为 `sdpa`；
2. actor 缺 `ppo_micro_batch_size_per_gpu` → config 补；
3. `rollout.tensor_model_parallel_size` 默认 2 → 设 1（单设备 PyTorch rollout）；
4. worker `generate_sequences` 读 `self.generation_config` → 覆写里设 `None`；
5. **FSDP MixedPrecision dtype**：actor 建 fp32 而 rollout 调子方法绕过 FSDP forward hook（bf16 cast 不生效）→ fc.weight fp32 vs 输入 bf16 → **actor 改为 bf16 构建**（与 DeepSpec 训练 + smoke 一致，统一 bf16）；
6. attn q/k/v dtype 不一致（autocast 副作用）→ 同 #5 统一 bf16 后消除。

**通过标准**：smoke 全绿（含 eval 一致性）+ E2E 构建真实 FSDP actor 并产出合理 rollout、rc=0 —— **单采样(R=1)路径已达成**。🟡 **S2 完整完成还需**：加 `R` 维多采样 + top-K 候选输出（见本节顶部状态框 + 张量契约 S2）。
> 🔷 **补充验证（可选，未做）**：裸模型 vs FSDP actor 的 logits 严格对拍（张量契约 S2 的 FSDP 透明性项）。当前 FSDP actor 用与 smoke 完全相同的 bf16+flex 配置且 `NO_SHARD` 无分片（每卡完整参数），数值一致性有强保证；如需严格证明可补一个对拍脚本。

---

### S3 — Teacher 打分：路线 A 块对角前向（把 IP-4 桩变真）

**目标**：teacher 块对角打分——常驻 target，对 rollout 用 **A×blk 展平 + 块对角因果 mask + 真实 position_id** 一次 online 前向，得 `logp_target_on_topk [B,A,blk,K]`（在**学生 top-K 候选**上，条件各块采样前缀）。**展平做法已锁定（§2.4.1），S3 须先验证其等价性再用。**

> **top-k 候选来自 S2（无 S3 学生前向）**：`student_top_k_ids` 在 S2 rollout 顺手 `topk(corrected_logits, K)` 取（DSpark 省了 Rethink-OPD 的 `compute_log_prob(top_k)` 那次独立前向，见张量契约「前向次数账」）；S3 直接消费。补齐 S2 的 top-k 输出属 S2 收尾，非 S3。

**交付物** ✅ 已实现
- （S2 收尾）`block_rollout` 补 `student_top_k_ids/logp = topk(corrected_logits, K)`（no_grad）。✅
- `recipe/dspark_opd/teacher_scoring.py`：`score_blocks_flat`（**展平块对角因果打分**，一次 target forward）+ `score_blocks_reference`（逐 block 独立前向，等价性 ground-truth）→ `logp_target_on_topk [B,A,blk,K]`。✅
- `recipe/dspark_opd/worker.py`：`DSparkRewardModelWorker._build_model`（构建 target，sdpa）+ **覆写整个 `compute_rm_score`**（**非** verl 的 `_forward_micro_batch`——后者耦合 Rethink-OPD 的序列级布局 `old_log_probs/responses`，与块对角结构不符；故与 actor 一样整体覆写，`@register(mesh_name="reward")` 重挂），内部调 `score_blocks_flat`。✅
- `recipe/dspark_opd/trainer.py`：teacher phase（拼 `input_ids` + rollout 块张量 → `rm_wg.compute_rm_score` → gate=`teacher`）。✅
- `scripts/opd/s3_smoke.py`（孤立校验，含展平等价性）。✅

**smoke-test（我执行，自动断言）** ✅ 通过（实测 num_anchors=8，2 样本 real_len=[226,244]）

```bash
PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec:/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl \
~/.venv/dspark-opd/bin/python scripts/opd/s3_smoke.py \
    --draft /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest \
    --target Qwen/Qwen3-4B --cache /mnt/scratch/qwen3_4b_target_cache --num-anchors 8 --seed 42
```
断言（见张量契约 S3 必做检查）：
- **★ 展平等价性（黄金校验，先验证再用）**：`score_blocks_flat` vs `score_blocks_reference`（逐 block 独立前向）逐 block `logp_target` 一致。**fp32 下 all-K `max|Δ|=1.45e-4`（top-1 `max=7.4e-5`）——结构上等价**；bf16 argmax 100% + top-1 & student-weighted `mean|Δ|≈0.02`（长展平序列的 bf16 精度噪声，与 S2 flex-vs-sdpa 同量级）。✅
  - **⚠️ 曾发现并修复一个真实 bug**：`allow_ctx` 原用 `key_pos <= anchor`，使 anchor token 被**双重计入**（既作 ctx key、又作 block-key `j=0`），bf16 下 top-1 `max|Δ|` 一度到 **7.0**。改为 `key_pos < anchor`（anchor token 只由 block 侧提供）后，fp32 残差降到 1e-4——**证明是 mask 构造错误而非精度噪声**，也印证了「展平前先验证等价性」的必要。
- **块对角隔离**：改某 block 采样 token，其它 block `logp_target` 不变。✅
- **块内因果**：改块内最后一个 token 不影响更早位置的 `logp_target`。✅
- `logp_target` `<=0`、有限。✅

> **k=1 黄金对拍暂缓**：与缓存 `target_last_hidden_states[anchor]` 过 lm_head 对拍需 anchor 位置的 target 分布；当前 route A online 前向已由「展平 vs 逐 block reference」等价性 + 块对角隔离/块内因果三项共同锁定正确性，k=1 缓存自洽对拍可在 S4 接入缓存 last_hidden 时补做（非 S3 阻塞项）。

**E2E test（你执行，人工检查）** — 统一命令（§3.0.3），gate=`teacher`（target 真实打分）：

> *（历史：原为 gated `STOP_AFTER=teacher`；现融合去 gate，命令如下）*

```bash
bash third_party/verl/recipe/dspark_opd/run.sh
```

**你应看到**：dataflow 走完 data→rollout→teacher（`compute_rm_score`）；trainer teacher phase 打印一张 **target vs draft 逐候选对照表**（sample 0 第一个有效 block 的 pos j=0，前 8 个 top-k 候选：`logp_target`、`logp_draft`、`Δ=T−D`——Δ>0 表 teacher 比 draft 更看好该候选，即 OPD 要拉近的方向）；gate=`teacher` 处 dump `logp_target_on_topk [B,A,blk,K]`（`<=0`、有限）+ `logp_draft_on_topk`（同候选、来自 S2）；末行 `[gate] phase=teacher, stopping cleanly (rc=0)`。

**通过标准**：smoke 展平等价性 + 块对角隔离 + 块内因果成立（证明打分正确、块对角 mask 无跨块泄漏）；E2E 打印合理的 `logp_target`/`Δ` 且 rc=0（证明 target 在真实 Ray/FSDP 装配下完成块对角 online 打分）。

---

### S4 — Reward→Advantage→Loss：训练前向 + 4D→3D 展平 + loss 拼接

**目标**：本阶段两件事（actor model 已在 S2 构建好，此处复用它做**训练前向**）——(a) **带梯度训练前向，经 `FSDP.forward`**：复用 S2 的 actor，通过 `Qwen3DSparkModel.forward`（加 OPD 分支）在**固定 anchor + 固定采样 token** 上复现 rollout 的 `corrected_logits`（`draft_logits`）+ `confidence_pred`（带梯度），再在**固定的 S2 top-k 候选 id** 上出带梯度学生 logπ；(b) 完成**最高风险项**——把 `(B, A, blk)` 的 4D block 结构**展平进 `(B, response_len)` 序列轴**，喂给 fork 现成的 `token_reward_direct` + 3D `compute_policy_loss_vanilla`，`eval_mask` 作 `response_mask`，并加回 confidence BCE。gate=`loss`：完成一次 `backward` + optimizer step 后 dump + `exit(0)`。

> **★【关键设计】训练前向必须经 `FSDP.forward`（即 `module(...)`），不得绕过。** FSDP1 的跨 rank 梯度归约 hook 在 `FSDP.forward` 的 pre-forward 阶段注册（verl 走 `self.actor_module(...)`，`dp_actor.py:175/346`）。若用 `model._forward_backbone(...)` 等**子方法直接调用**绕过 `FSDP.__call__`，则 root flat-param 里的可训练 OPD 头（`fc`/`hidden_norm`/`norm`/`markov_head`/`confidence_head`）的梯度**在多卡上永不 all-reduce → 副本静默发散**（decoder layers 因 `_no_split_modules` 单独包裹、其 `__call__` 在 layer 循环触发，梯度会同步；受害的是 root-unit 训练头）。故 `update_dspark_opd` 调 `module(input_ids=…, anchor_positions=固定, block_prev_tokens=采样)`，让 `Qwen3DSparkModel.forward` 走 OPD 分支。
> - **`Qwen3DSparkModel.forward` 加两个可选参**：`anchor_positions=None`（None → 现状 `sample_anchor_positions`，**SFT 路径零改动**；否则用传入固定 anchor）、`block_prev_tokens=None`（None → 现状 `[anchor_tok, target_ids_{<blk}]` 真实 token teacher-force；否则用传入采样 token）。`draft_logits`（= `compute_logits`+`markov apply_block_logits`）本就等于 OPD 要的 `corrected_logits`，**无需新增输出字段**。
> - **梯度同步需两层同时成立**：① 训练前向走 `FSDP.forward`（本节，注册 post-backward hook）；② actor 用 **`NO_SHARD`** 而非 `fsdp_size=1` 的退化 HYBRID mesh（§2.6.5，HYBRID(1) 的 replicate 维不 all-reduce）。缺任一层多卡梯度都不同步。`NO_SHARD` 下每卡完整参数，rollout 子方法调用仍安全（无需改 rollout）。
>
> > **（B）修订说明**：S4 初版曾把这段前向写成独立函数 `dspark_block_train_forward`（用 `model._forward_backbone(...)` 子方法调用，绕过 FSDP），单卡 smoke 全过但**多卡梯度不同步**（5-agent review 发现的 CRITICAL，详见 memory `dspark-opd-multigpu-grad-sync-bug`）。已按上述设计修正为走 `FSDP.forward`。

> **【已锁定】confidence 目标用 top-K 支撑近似**：`accept_rate = 1 − ½·Σ_V|p_draft − p_target| = Σ_V min(p_draft, p_target)`，**限制在学生 top-k 支撑上** `≈ Σ_k min(p_draft_k, p_target_k)`——直接复用 S2 的 draft top-k logp + S3 的 teacher top-k logp（**均 no-grad，零新增跨 worker 数据搬运**）。是全词表 TV 的近似（忽略 top-k 尾部质量，通常 top-k 已覆盖绝大部分概率）；detach 后作 BCE 目标。若 eval 显示 confidence 退化可再升级为全词表精确（须把 draft `corrected_logits [B,A,blk,V]` 搬进 reward worker）。
>
> **【落地决策】不走 verl 的 `compute_log_prob→compute_distillation_reward→update_policy` 三段式**（均耦合序列级 CausalLM 布局），而与 rollout/teacher 一致**整体新增一个 actor 方法 `update_dspark_opd`**（`@register(mesh_name="actor")`），一次做完 grad forward（经 `FSDP.forward`）→reward→loss→backward→step。内部复用 verl 现成的 `token_reward_direct` + 3D `compute_policy_loss_vanilla` + actor `_optimizer_step`（grad-clip/step）。
>
> **【micro-batch 切分 + 梯度累积（照搬 verl 静态分支）】** `update_dspark_opd` 按 `ppo_micro_batch_size_per_gpu` 把本卡 batch `data.split()` 成若干 micro，逐 micro 经 `module(...)`+reward+PG+conf loss、`(loss × 1/n_micro).backward()` 累积梯度，末尾一次 `_optimizer_step`。这与 verl `update_policy`（`dp_actor.py:801-819`）**语义一致**：worker 收到的整批 = 一个归一化后的 mini（trainer 已 repeat `rollout.n`、dispatch 按 dp 切），`n_micro = B_recv // ppo_micro_batch_size_per_gpu` 恰等于 verl 的 `gradient_accumulation`。
> - **聚合方式**：每 micro 内部各自 `token-mean`，micro 间用固定系数 `1/n_micro` 聚合。各 micro 有效 token 数不等时与"整批一次 token-mean"有偏差（verl 主线同款近似，靠 `balance_batch` 缓解，见 §S6 前置项②/③）——**刻意保持与 verl 一致**，不做 `D_i/ΣD_i` 补偿。
> - **⚠️ 与 Rethink-OPD 差异（已知、刻意）**：Rethink-OPD OPD 全程 `use_dynamic_bsz=True`（`loss_scale=样本数/mini`，按 token 打包处理变长 response）；我们对齐 verl **静态分支**（`1/n_micro` 等权）。因块结构 per-sample 计算量固定，静态切分下各 micro 样本数相等，`1/n_micro` 与 dynamic 的 `样本数/mini` 数值一致——`ppo_micro_batch_size_per_gpu=1` 时每 micro 恰 1 样本、`n_micro=B_recv=mini`，两式**严格相等**；仅 `micro>1` 且 B 不整除时末尾 micro 权重略偏。dynamic 对我们收益小、复杂度高，故维持静态。
> - **性质**：解决多卡/大 `num_anchors` 的**显存**（整批一次全词表前向会 OOM），config 的 `ppo_mini_batch_size`/`ppo_micro_batch_size_per_gpu` 至此**真正生效**。

**交付物**（🔨 前向重构中；reward/loss/展平/micro-batch 逻辑已实现并单卡验证）
- `deepspec/modeling/dspark/qwen3/modeling.py`：`Qwen3DSparkModel.forward` 加可选参 `anchor_positions`/`block_prev_tokens`（OPD 分支；默认 None → SFT 路径零改动）。gemma4 对应模型若支持 OPD 同步改。🔨
- `recipe/dspark_opd/worker.py`：`DSparkActorRolloutRefWorker.update_dspark_opd`（`@register(mesh_name="actor")`）——micro 循环内改调 `module(...)`（`FSDP.forward`，OPD 分支），从 `DSparkForwardOutput` 取 `draft_logits`(=corrected)/`confidence_pred`/`eval_mask`；`block_prev_tokens=cat([anchor_tok, rollout_tokens[:,:,:-1]])`。🔨
- `recipe/dspark_opd/loss_bridge.py`（**纯函数**）：`logp_on_topk_ids` + `build_opd_reward`（`rm=w·(T−S)`）+ `flatten_blocks_to_sequence`/`unflatten_sequence_to_blocks` + `confidence_accept_rate_topk` + `confidence_bce` + `block_decay_weight_mask`。✅（`dspark_block_train_forward` 移除——逻辑回归 `forward`）
- `recipe/dspark_opd/trainer.py`：loss phase（拼 data+rollout+teacher → `actor_rollout_wg.update_dspark_opd` → gate=`loss`）。✅
- `recipe/dspark_opd/config/dspark_trainer.yaml`：`override_config` 补 `loss_decay_gamma:4.0`、`confidence_head_alpha:1.0`。✅
- `scripts/opd/s4_smoke.py`（单卡：展平/reward/PG/backward/micro 累积）+ `scripts/opd/s4_grad_sync_smoke.py`（**多卡梯度一致性**，见下）。🔨

**smoke-test（我执行，自动断言）**

**① 单卡（`s4_smoke.py`，实测 num_anchors=8，2 样本）**
```bash
PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec:/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl \
~/.venv/dspark-opd/bin/python scripts/opd/s4_smoke.py \
    --draft /mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest \
    --target Qwen/Qwen3-4B --cache /mnt/scratch/qwen3_4b_target_cache --num-anchors 8 --seed 42
```
断言：
- **★ 展平可逆性（黄金）**：`unflatten(flatten(x)) == x`（K 张量 + `eval_mask` 逐元素相等）——防 reward 错位到别的 block 位置。
- **前向经 `model(...)`（OPD 分支）自洽**：`model(input_ids, …, anchor_positions=固定, block_prev_tokens=采样)` 的 `draft_logits`(=corrected) 上取固定候选带梯度 logπ，对比 S2 no-grad `student_top_k_logp` `mean|Δ|≈0`（backbone 确定 + markov teacher-force 精确复现）。**（改造校验）改造后 `model(...)` 的 corrected_logits 与旧 `dspark_block_train_forward` 逐元素 `allclose`**（若保留旧函数作对照）。
- **SFT 零回归**：`model(input_ids, target_hidden_states, loss_mask, target_last_hidden_states)`（**不传**新参）输出与改造前**逐元素相同**（默认 None 分支 == 原 `forward`）。
- **reward**：`rm=w·(T−S)` 有限、`eval_mask` 外为 0；`token_reward_direct` 的 `advantage==rm*mask`、`returns==adv`。
- **3D dual-clip PG**：`compute_policy_loss_vanilla` 3D 分支返回有限标量 `pg_loss`；on-policy `ppo_kl≈0`。
- **backward**：`accept_rate∈[0,1]`、confidence BCE 有限；可训练参数（backbone/`fc`/markov/confidence）grad `isfinite` 且 grad_norm>0，冻结 `embed_tokens`/`lm_head` 无梯度。
- **micro-batch 累积等价（test F）**：duplicate 出分母相等的 2 样本，「整批一次 backward」vs「2 micro 各 `loss×1/2` 累积」梯度 `max|Δgrad|<1e-2`。

**② ★ 多卡梯度一致性（黄金，`scripts/opd/s4_grad_sync_smoke.py`，本次修复核心）**
```bash
PYTHONPATH=…:…/third_party/verl torchrun --nproc_per_node=2 \
    scripts/opd/s4_grad_sync_smoke.py --draft … --cache …
```
`torchrun --nproc_per_node=2` 起 2 rank，FSDP 包裹 draft（`NO_SHARD`，同 worker 配置），各喂**不同**数据分片，走 `module(...)`（`FSDP.forward`）+ backward + 一次 `SGD.step`，然后 **`summon_full_params` 后 all-gather 各 rank 的 root-unit 训练头参数（`fc`/`markov_head`/`confidence_head`/norms）比对 → step 后必须 `allclose`**。用**"step 后参数一致"而非"原始 grad 一致"**判定：`use_orig_params`/分片下 `.grad` 视图可能是归约前的、易误判；而两 rank 起点相同（`sync_module_states`）→ 若梯度已 all-reduce 则 step 后参数仍一致，否则（各喂不同数据）发散。**这是之前缺失的、专门覆盖多卡梯度同步的 smoke**；对照实测已确认 `NO_SHARD` `Δparam=0`（同步）、`HYBRID(fsdp_size=1)` `Δparam≈2`（不同步）。

**E2E test（你执行，人工检查）** — 统一命令，gate=`loss`：

> *（历史：原为 gated `STOP_AFTER=loss`；现融合去 gate，命令如下）*

```bash
# 单卡（含一次 backward，验证全链路）
bash third_party/verl/recipe/dspark_opd/run.sh
# 多卡（验证梯度同步 + 无 hang）
NGPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash third_party/verl/recipe/dspark_opd/run.sh
```

**你应看到**：完整 dataflow data→rollout→teacher→`update_dspark_opd`（经 `FSDP.forward` 的 grad forward→reward→PG+confidence loss→一次 `backward`+optimizer step）全部真实执行，gate=`loss` 处打印标量 `actor/loss`、`actor/pg_loss`、`actor/confidence_loss`、`actor/grad_norm`（>0）、`actor/ppo_kl`（≈0）；多卡下无 hang；末行 `[gate] phase=loss, stopping cleanly (rc=0)`。

**通过标准**：单卡 smoke 全过 + **多卡梯度一致性 `allclose`** + E2E（单/多卡）rc=0。**多卡梯度一致性是本阶段核心**——决定「能否复用 fork 3D 管线」且「多卡训练数值是否可信」。

---

### S5 — 去桩、放开完整单 step（gate=off）✅（代码实现）

**目标**：此时四个 IP 已全部变真（S1–S4），本阶段**去掉 gate**（`stop_after=off`），跑一个**完整 optimizer step**（含 optimizer.step + 落盘一次），验证「多步连跑」的编排闭环无残留问题。无新模块，纯集成收尾。

> **⚠️【架构修正，重要】`stop_after=off` 不委托 `super().fit()`，而是走我们自建的多 step 循环。** 原计划写"委托 stock `RayPPOTrainer.fit`"——**这是错的**：stock `fit()` 硬编码序列级 CausalLM 管线——① `_get_gen_batch`（`ray_trainer.py:523`）只 pop `input_ids/attention_mask/position_ids`，**丢掉我们 rollout 必需的 `target_hidden_states/loss_mask`**；② 调 `compute_log_prob`→`compute_distillation_reward`→`update_actor` 三段式，而我们实现的是 `generate_sequences`/`compute_rm_score`/`update_dspark_opd`（S2–S4 一贯的整体覆写风格），stock 链**根本不会调到**它们。故 S5 把 gated trimmed 循环重构为：per-batch pipeline 抽到 `_run_opd_step`（S1–S4 gated 与 S5+ full 共用），`off` 时去 gate + 跑 `total_training_steps` 的多 step 外循环。
>
> **repeat 行为**：S1–S4 gated 不做 `repeat`（batch 维不膨胀，如 `(2,32,7)`，便于检查）；S5+ 在 `_prepare_batch` 里**自己做** `batch.repeat(rollout.n)`（对应 stock 的 `ray_trainer.py:1042`，我们不走 stock 故自行实现），batch 从 `B_prompt=2` 膨胀为 `B_prompt×n=8`，每个副本在 `generate_sequences` 里独立采 anchor（`sample_anchor_positions` 逐行 `torch.rand`，R-in-batch，张量契约 §S2）。**这是 rollout 次数 `n` 真正在 batch 维落地的地方**——S5 会看到 batch ×4，属预期。

#### `DSparkTrainer.fit()` vs `RayPPOTrainer.fit()` 逐项对比

> **澄清"用哪个 Trainer"**：我们**从未**直接实例化/运行 stock `RayPPOTrainer`。自 S0 起就一直是 `DSparkTrainer(RayPPOTrainer)`——**继承**父类的 `__init__`/`init_workers`/`_save_checkpoint` 等**装配与落盘基础设施**，但从 S0 起就**覆写 `fit()`** 用自建 gated 循环，从未调用父类 `fit()` 的循环体。变化的不是"Trainer 类"（一直是 `DSparkTrainer`），而是"`off` 时 `fit()` 内部是否委托 `super().fit()`"——本次由"是"（原计划）修正为"否"。
>
> 两者**外壳骨架相同**（epoch → dataloader → 每 batch 一步 → 按 `save_freq` checkpoint → `global_steps` 推进到 `total_training_steps`），差异全在**每 batch 内部的 pipeline**：

| 环节 | `RayPPOTrainer.fit()`（父类 stock） | `DSparkTrainer.fit()`（我们，S5 定稿） |
|---|---|---|
| **gen batch 构造** | `_get_gen_batch`（`ray_trainer.py:523`）`pop` 出 `input_ids/attention_mask/position_ids`，**丢弃其余字段** | 整个 batch（含 `target_hidden_states`/`loss_mask`）直接传给 rollout worker |
| **rollout.n repeat** | `gen_batch.repeat(n)` + `batch.repeat(n)`（`:1042/:1090`） | `_prepare_batch` 里 `batch.repeat(n)`（自己做，语义相同） |
| **rollout 调用** | `generate_sequences`（vLLM / async manager），产出 `responses`（序列） | `generate_sequences`（我们的 `DSparkRollout`），产出 `rollout_tokens [B,A,blk]` 等块结构 |
| **student logp** | 独立一次 `compute_log_prob`（vLLM 不留 logits，须补算） | **无此步**——top-k logp 在 rollout 顺手出（省一次前向，见「前向次数账」） |
| **teacher 打分** | `compute_rm_score`——序列级 `[B,T,K]` | `compute_rm_score`——块对角 `[B,A,blk,K]`（route A） |
| **reward 组装** | `compute_distillation_reward`（独立 worker 调用） | **合并进** `update_dspark_opd` 内部（不单独暴露） |
| **advantage** | 驱动进程 `compute_advantage(...)`（独立步骤） | 合并进 `update_dspark_opd`（`token_reward_direct` 内联） |
| **actor 更新** | `update_actor`→`DataParallelPPOActor.update_policy`（序列级 + micro-batch 切分 + 梯度累积 + `ppo_epochs` 循环） | `update_dspark_opd`——grad forward + 3D PG + confidence BCE；**micro-batch 切分 + 梯度累积（`1/n_micro`，照搬 verl）+ 末尾一次 step**，见 §S4 note |
| **confidence 头** | 无（verl 无此概念） | 有——DSpark 专有 BCE（top-k 近似目标） |
| **response_mask** | `compute_response_mask` 从 `attention_mask` 派生 | `eval_mask × decay` 派生（张量契约 §S4） |
| **critic / ref_policy / reward_fn / KL penalty** | 完整支持（GAE critic、reference policy、rule-based reward、`apply_kl_penalty`） | **全部不走**（OPD 无这些：`use_critic=False`、无 ref、reward 即 teacher KL） |
| **validate / balance_batch / 可视化** | 有（`_validate`、`_balance_batch`、swanlab 候选分布画图等） | 无（精简掉） |
| **checkpoint** | `_save_checkpoint()`（按 `save_freq`） | **复用父类 `_save_checkpoint()`**（同一方法） |
| **多 step 外循环** | epoch × dataloader，`global_steps` 推进 | **结构相同**（epoch × dataloader，到 `total_training_steps` 停） |

> **不能委托 `super().fit()` 的根因**：父类链的**第一步 `_get_gen_batch` 就丢掉 `target_hidden_states`**（我们 rollout 必需），且它调的 `compute_log_prob`/`compute_distillation_reward`/`update_actor` 我们**都没实现**（我们实现的是 `generate_sequences`/`compute_rm_score`/`update_dspark_opd`）。故继承父类只为复用装配/落盘基础设施，训练循环必须自建。

**交付物** ✅ 已实现
- `recipe/dspark_opd/trainer.py`：`fit` 重构（`off` 走自建多 step 循环，非 `super().fit()`）+ `_prepare_batch`（rollout.n repeat + uid）+ `_run_opd_step`（gated/full 共用的 per-batch pipeline）+ 按 `save_freq` 调 `_save_checkpoint`（复用父类 `FSDPCheckpointManager`）。
- `recipe/dspark_opd/config/dspark_trainer.yaml`：`trainer` 补 `save_freq:1`、`total_training_steps:1`、`total_epochs:100`（小数据集循环到步数为止）。

> S5 已无独立 smoke（模块正确性由 S1–S4 smoke 覆盖 + S5 纯控制流重构，per-batch pipeline 未变，已在 gate=`loss` E2E 验证）；仅 E2E。

**E2E test（你执行，人工检查）** — 统一命令（§3.0.3），跑 1 个完整 step：

> *（历史：原为 gated `STOP_AFTER=off`；现融合去 gate，命令如下）*

```bash
STEPS=1 SAVE_FREQ=1 bash third_party/verl/recipe/dspark_opd/run.sh
# （STEPS/SAVE_FREQ 控制 total_training_steps/save_freq，S5 设为 1 验证单步闭环）
```

**你应看到**：batch 膨胀为 `B_prompt×n`（2×4=8），1 个完整 step 正常结束——日志 `[DSparkTrainer] step 1/1 actor/loss=… actor/pg_loss=… actor/grad_norm=…`（`grad_norm>0`、无 NaN），step 1 落盘一个 checkpoint（`checkpoints/dspark_opd/<exp>/global_step_1/actor`），末行 `[DSparkTrainer] full fit done at step 1`，进程 rc=0。

**通过标准**：单完整 step 无报错、checkpoint 落盘、rc=0。**常见失败**：optimizer/FSDP 状态、checkpoint 落盘路径、`rollout.n` repeat 后显存（B×n×A 个 block 的 target 打分 + 学生带梯度前向）。

---

### S6 — 端到端小规模训练跑通（多卡、多 step）

**目标**：多卡、数百 step 的短训练，验证 loss 下降趋势与训练稳定性（不追求最终指标）。仍是同一 verl 入口，`stop_after=off` + 多 step + 多卡。

> **S6 前置项（多卡正确性，5-agent review 结论）**：
> 0. **★【CRITICAL】训练前向走 `FSDP.forward`（梯度跨 rank 同步）** → **S4 设计已含**（见 §S4「训练前向经 `FSDP.forward`」）。这是 S6 多卡训练数值可信的**硬前提**，不走 `FSDP.forward` 则 4 副本静默发散。
> 1. **micro-batch 切分 + 梯度累积** ✅ **已补齐**（见 §S4 note）：`update_dspark_opd` 按 `ppo_micro_batch_size_per_gpu` 切本卡 batch、`loss×1/n_micro` 累积、末尾 `_optimizer_step`。解决显存、与 verl `update_policy` 语义一致。
> 2. **跨卡 token-mean 归一化偏差** 🔷 **待观察**（多卡专属，S5 单卡不触发）：PG 用 `token-mean`（loss÷本卡有效 token），FSDP HYBRID_SHARD 又对梯度按卡数简单平均；各卡 `eval_mask` 有效 block 数不等时，等效"先各卡内均值再等权平均" ≠ "全局所有 token 一起均值"。**这与前置项①里 micro 间 `1/n_micro` 聚合是同源偏差**。verl 主线同款、靠 `balance_batch` 缓解。**我们已选择照搬 verl（接受此近似），S6 默认不动，仅实测偏差显著时再处理。**
> 3. **`_balance_batch` 未调用** 🔷（verl `trainer.balance_batch` 默认 True）：`DSparkTrainer.fit` 未调，各卡 token 不均 → 负载失衡 + 加剧②的归一偏差。S6 可考虑接入（trainer 里 update 前调 `self._balance_batch(batch, metrics)`）。
> 4. **无 `_load_checkpoint` / resume** 🔷：`fit` 不调 `_load_checkpoint`、`global_steps` 恒从 0——多 step 重启会从头训并覆盖旧 ckpt。S6 若需断点续训须补（verl `ray_trainer.py:988`）。
> 5. **metrics 只取 rank0（未跨 DP 归约）** 🔷：`update_dspark_opd` 返回 shape-[1] 张量经 collect 拼成 [world]，trainer 只读 `[0]`（`trainer.py`）。多卡下日志失真（不影响训练）。可在 trainer 端对 metrics 求均值。
> 6. **reward worker `reward_module._handle.reshard(True)`** ✅ **已补**（`compute_rm_score` 末尾，`world_size>1 and fsdp_version==1`，镜像 verl `fsdp_workers.py:2753`）：teacher（`fsdp_size=-1`→FULL_SHARD + CPUOffload）前向后 reshard root 参数，避免多卡显存膨胀/OOM。第二轮 review 发现原注释误称"fsdp_size=1 无需 reshard"（错误前提），已一并修正。
> 7. **dispatch 整除要求** ⚠️：非 padding chunk 断言 `B % world_size == 0`。S6 `train_batch_size×n=8` 对 4 卡 OK（`B=train_batch_size×n`，`n=world=4` → 恒成立）；gated 路径 `B=2` 对 4 卡会触发断言（gated 实际单卡，潜在）。
> 8. **checkpoint 可移植性** 🔷（第二轮新增）：inherited `_save_checkpoint`→`FSDPCheckpointManager` 用 `SHARDED_STATE_DICT`，文件名含 `world_size_{ws}_rank_{rank}`。NO_SHARD 下每卡存**完整副本**（磁盘×world）且**换 GPU 数不能 resume**。实测 save+reload 同 world 下 OK。属运维限制，非正确性问题；与④(无 resume)一并考虑。
> 9. **offload 未接线** 🔷（第二轮确认）：`update_dspark_opd` 未按 `_is_offload_param/_optimizer` 做 load/offload（verl `fsdp_workers.py:869-906`）。当前 offload 默认关，无害；**若开启则 param 在 CPU 而 forward 输入在 GPU → 设备不匹配崩溃**（非静默）。开 offload 须一并补 load/offload。

> **第二轮 review 已澄清为「安全/非问题」**（源码 + 实测确认，不必处理）：① NO_SHARD 下 `FSDP.clip_grad_norm_` 走本地范数分支，但因 post-backward hook 已 all-reduce 平均梯度（`_reduce_grad_no_shard`），clip 前各卡 `.grad` 一致 → 裁剪一致（实测 grad_norm Δ=0）；② `grad_clip` 默认 1.0（`_optimizer_step` 断言不崩）；③ `device_mesh=None` + `fsdp_size=-1` 的 1D mesh 使 `ppo_mini_batch_size` 归一化正确；④ rollout 子方法在 NO_SHARD 下看到完整参数（`set_module` 用未包装 module）；⑤ 无 partial-rank 集合通信 → 跳过 validate 不 hang。

**交付物**：`run.sh` 的多卡配置 + 一次短训练日志 + tensorboard（micro-batch/梯度累积 + 训练前向走 `FSDP.forward` + `NO_SHARD` 梯度同步 + reward reshard 均在 S4 落地）。

> S6 无新模块；小改动：`dataloader_num_workers:0`（teardown 噪声）+ run.sh 加 S6 旋钮。正确性修复集中在 S4 + 前置项①⑥已补；前置项 ②③④⑧⑨ 按实测需要择机接入（均非训练数学错误）。2-GPU 3-step 已 de-risk，full 多卡长跑你执行。

**已 de-risk（我执行，2-GPU 3-step 短跑实测）** ✅：`STOP_AFTER=off NGPUS=2 STEPS=3 SAVE_FREQ=3 CUDA_VISIBLE_DEVICES=0,1`：
- 3 step 在 2 卡上跑完（rank0/rank1 各自 rollout+train），loss 有限、`grad_norm` 有限有界（2.0→6.3→3.0）、`ppo_kl=0`（on-policy），**无 NaN、无发散**。
- step 3 双卡各落盘 `model/optim/extra_state`（NO_SHARD 每卡完整副本 `model_world_size_2_rank_{0,1}.pt` 各 2.78GB），`[DSparkTrainer] full fit done`，rc=0。
- **修了一个 teardown 噪声**：`dataloader_num_workers` verl 默认 8 → 训练结束 `fit()` teardown 时持久 DataLoader worker 被 SIGKILL，冒出一条**训练完成后**的 `DataLoader worker killed` traceback（rc 仍 0、checkpoint 完整、非 OOM——主机 1.9TB 空闲）。设 `dataloader_num_workers: 0`（主进程加载，我们数据集仅 ~8 条缓存张量，零吞吐损失）后噪声消失。

**run.sh 新增 S6 旋钮**：`STEPS`（`total_training_steps`）、`SAVE_FREQ`、`EXP`（`experiment_name`）。

**E2E test（你执行，人工检查）** — 统一命令，多卡多 step：

> *（历史：原为 gated `STOP_AFTER=off`；现融合去 gate，命令如下）*

```bash
NGPUS=4 STEPS=200 SAVE_FREQ=100 EXP=s6_run CUDA_VISIBLE_DEVICES=0,1,2,3 \
    bash third_party/verl/recipe/dspark_opd/run.sh
```

> **多卡日志形态（必知，非 bug）**：verl 走 **Ray**（非 torchrun）。`run.sh` 起的是单个 **driver** 进程，`DSparkTrainer.fit()` 在 driver 上跑 → `[DSparkTrainer] step ...` 等**天然单流**；真正的 N 个 GPUworker 是 Ray remote actor，其日志被 Ray 加 `(WorkerDict pid=...)` 前缀转发（可能去重）。验证多卡在跑：① `nvidia-smi` 看各卡进程；② `/tmp/ray/session_*/logs/` 看 per-worker 日志（含 `RANK/WORLD_SIZE`）。

**你应看到 / 通过标准**
- 200 step 无 OOM、无 NaN（日志无 `grad_norm is not finite`）。
- OPD 反向 KL loss（`teacher_logp − draft_logp` 相关项）**总体下降**；`grad_norm` 稳定不爆。
- step 100 + 200 各落盘 checkpoint。
- 吞吐（step/s）记录在案（路线 A 有 target 常驻 + 每步打分，预期慢于纯缓存训练——量化它）。
- ⚠️ **NO_SHARD checkpoint 不可跨卡数 resume**（文件名含 `world_size`，每卡全量副本）——S6 若中断重启须用**相同 NGPUS**（且当前无 `_load_checkpoint`，见前置项④，重启从头训）。

---

### S7 — checkpoint 转换 + `eval.sh` 对照基线（§2.7）

**目标**：把 S6 的 verl checkpoint 转成 `Qwen3DSparkModel.from_pretrained` 可读格式，用**未改动的** `scripts/eval/eval.sh` 评测，与现有 L1/CE 基线（`step_latest`）对比接受率/接受长度。

**交付物**
- `scripts/opd/convert_ckpt.py`：verl FSDP checkpoint → HF-style draft checkpoint 目录（config.json + safetensors）。
- 一张对照表（OPD vs 基线）。

**smoke-test（我执行，自动断言）** — 转换保真：脚本转换后 `Qwen3DSparkModel.from_pretrained` 成功加载，`block_size==7`、layers `==[1,9,17,25,33]`，且与 verl checkpoint 逐张量 `allclose`。

**E2E test（你执行，人工检查）**

```bash
# 1) 转换
PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec \
~/.venv/dspark-opd/bin/python scripts/opd/convert_ckpt.py \
    --verl-ckpt <S6 checkpoint 路径> \
    --out /mnt/scratch/checkpoints/deepspec/dspark_opd_qwen3_4b/step_200

# 2) 评测：完全复用 eval.sh（仅把 draft_name_or_path 指向 step_200，脚本本体零改动）
bash scripts/eval/eval.sh 2>&1 | tee /tmp/s7_eval_opd.log
```

**你应看到 / 通过标准**
- `eval.py` 用 `EVALUATORS["Qwen3DSparkModel"]` 正常加载评测，打印接受率表（`accept_rate@k`、`acceptance_length`、`verify_rate`）。
- 与基线 `step_latest` 同口径对比：得到 OPD vs L1/CE 的**接受长度差值**（本项目的最终成败指标）。
- `eval.sh` **零改动**跑通。**成功判据（实验目标）**：OPD checkpoint 的 `acceptance_length` ≥ 基线（理想更高，尤其 `accept_rate@k` 在 k≥2 处提升，印证 §2.3「on-policy 主要改善 markov/confidence 头的 exposure bias」）。

---

### 3.1 里程碑与风险登记

| 里程碑 | 完成标志 | 主要风险（见对应阶段） |
|---|---|---|
| M0：集成地基就绪 | S0 通过 | **环境依赖冲突 + verl recipe 骨架/桩/gate 能启动**（S0，集成风险已前置到此） |
| M1：数据/rollout 就位 | S1–S2 通过 | float8 反量化（S1）；rollout 对拍（S2） |
| M2：核心算子正确 | S3–S4 通过 | k=1 对拍（S3）、**block→序列展平**（S4，最高算法风险） |
| M3：训练跑通 | S5–S6 通过 | 完整 step 编排、FSDP 包自定义模型、吞吐 |
| M4：指标验证 | S7 通过 | checkpoint 转换保真、接受长度是否超基线 |

**方法学要点（呼应本方案）**：verl 从 S0 接入 + 占位桩策略，把「环境冲突、verl 装配、Ray、FSDP 包自定义模型、DataProto 契约、gate 机制」这些最易返工的**集成风险一次性前置到 S0**；其后每阶段只替换一个桩，通过同一 verl 入口 + `stop_after` gate 在真实调用链里验证。因此**最深的算法风险（S4 的 4D→3D 展平）落地时，其上下游（rollout/teacher/verl 编排）已在 S0–S3 用真实或占位形式验证过**，出问题时容易定位到展平本身而非集成噪声。
