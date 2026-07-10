# 附录：基于 verl 0.8.0 承载 DSpark-OPD 的调研与设计（备选，未选中）

> 本文档是 [`DSpark-OPD.md`](./DSpark-OPD.md) 的附录，记录「用 **verl 0.8.0** 承载 DSpark-OPD」这条路线的完整调研与设计。
>
> **结论（2026-07-07）**：该路线**实现过重，不作为首选**。verl 0.8.0 是相对 0.7.0 的大规模架构重构，DSpark 的三处非标准点（非 vLLM rollout、非 CausalLM 草稿头 actor、块对角 teacher forward）恰好都落在 0.8.0 硬编码的「硬缝」上，均需 fork engine/worker 而非 plug-in。主文档已转向调研 **verl 0.7.0** 路线（见 `DSpark-OPD.md` §2.6）。
>
> 本文所有结论基于对 v0.8.0 tag `7aed6b2`（commit `7aed6b230776f963fa09509c10d9c3a767d1102c`，`version=0.8.0.dev`，tag 日期 2026-06-01）源码及其 `recipe/` 子模块（`verl-recipe@e7f8895`）的实读。文件路径 GitHub 形式为 `https://github.com/volcengine/verl/blob/v0.8.0/<path>`。

---

## A.1 verl 0.8.0 的原生 OPD（是什么、能否直接用）

verl 0.8.0 **相对 0.7.0 是大规模架构重构**。主文档 §1 剖析的 Rethink-OPD 是 **0.7.0 fork**，其 `workers/actor/dp_actor.py`、`workers/fsdp_workers.py`、`RewardModelWorker`、`token_reward_direct` advantage、以及「dual-clip PG 支持 3D top-k」等**在 0.8.0 中均已不存在或不成立**（全树 grep 零命中）。0.8.0 用新的 engine/worker 抽象（`workers/engine/`、`workers/engine_workers.py`）+ 实验性 loop 子系统（`teacher_loop`/`reward_loop`）替代。网上大量扩展文档针对的是 pre-0.8 布局，与此 tag 不符。

0.8.0 **核心内置了 first-class OPD**（不再只是 recipe）：
- 配置：`workers/config/distillation.py`（`DistillationConfig`：`enabled`、`teacher_models`、`teacher_key`、`distillation_loss{loss_mode, topk, use_policy_gradient, distillation_loss_coef, clip_ratio*, ...}`），YAML 默认在 `trainer/config/distillation/distillation.yaml`。入口仍是 `main_ppo` + `distillation.enabled=True`（`trainer/main_ppo.py:178-212` 分配 `teacher_pool` 并注册 `Role.TeacherModel`，actor loss 从 `ppo_loss` 切到 `distillation_ppo_loss`）。示例 `examples/on_policy_distillation_trainer/`，权威文档 `docs/algo/opd.md`。
- Loss（`trainer/distillation/losses.py`，`@register_distillation_loss` + `DISTILLATION_LOSS_REGISTRY`）：
  - `forward_kl_topk`（top-k 前向 KL，在 logits processor 里算，teacher 提供 top-k logprob，`use_topk=True`）；
  - `kl/k1/abs/mse/k2/low_var_kl/k3`（单样本 KL 估计，经 `core_algos.py` 的 `kl_penalty`，`use_estimator=True`）；
  - 两种应用模式：**PG-OPD**（`use_policy_gradient=True`，默认，把 `−distillation_loss` 当 reward 走 PPO policy loss，显式对齐 Thinking Machines OPD 博客）与 **监督 GKD**（`use_policy_gradient=False`，直接反传 KL，对齐 arXiv:2306.13649 Agarwal GKD）。
- Teacher（`experimental/teacher_loop/teacher_model.py`、`teacher_manager.py`）：`TeacherModelManager`/`MultiTeacherModelManager` 把 teacher 作为 **vLLM/SGLang 推理服务副本**拉到独立 Ray 资源池；`AsyncTeacherLLMServerManager.compute_teacher_logprobs_single()` 用 `prompt_logprobs`（top-k）对「学生 prompt+response」取 teacher logprob，写回 `teacher_ids`/`teacher_logprobs`。支持按 `teacher_key` 多 teacher 路由（MOPD）。

**关键约束**：这套原生 OPD **硬假设** student 与 teacher 都是**共享词表的标准 CausalLM**、teacher **可被 vLLM/SGLang 服务**、信号是**标准序列上的 top-k next-token logprob**。`DistillationTeacherModelConfig._validate_topk_logprobs` 只接受 `inference.name ∈ {vllm, sglang}`，否则 `NotImplementedError`。→ **DSpark 的块对角 target 打分无法表达为 vLLM `prompt_logprobs`，原生 teacher 路径无法承载。**

> 另注：`recipe/` 子模块另含一套独立的 Megatron OPD recipe `recipe/gkd/megatron/`（`kl`/`rkl`/`kl_rkl`/`jsd` on `teacher_topk_logps`+`teacher_topk_indices`，外部 ZeroMQ+vLLM teacher server，one/two-step-off 异步调度）。但每个 recipe 用 `REQUIRED_VERL.txt` pin 自己的 verl commit——gkd pin 的是 `bcb6386`（rolling main），非 v0.8.0 tag。故它是并行/替代实现，不是 0.8.0-native 的那套。

## A.2 四个集成点（IP）：哪些是「装饰器注册」，哪些必须 fork

| # | 集成点 | 0.8.0 现状（实读） | DSpark 处理 | 性质 |
|---|---|---|---|---|
| **IP-1** | Rollout | rollout 引擎经 `get_rollout_class(name,mode)` 选择（`engine_workers.py:608`），`_ROLLOUT_REGISTRY`（`workers/rollout/base.py`）只含 `(vllm/sglang/trtllm, async)`；`HFRollout`/`NaiveRollout`（`rollout/hf_rollout.py`、`rollout/naive/naive_rollout.py`）仍在且被导出，但**未接线**进 `ActorRolloutRefWorker`（worker 只调 `get_rollout_class`，无 `name=="hf"` 分支）。契约 `BaseRollout.generate_sequences(DataProto)->DataProto`，**刚性要求 prompt+response 序列**（2D `[B, resp_len]`，键 `prompts/responses/input_ids/attention_mask/position_ids/(rollout_)log_probs`） | 自研 `BaseRollout` 子类做块并行多-anchor 采样（复用 `deepspec/eval/dspark/draft_ops.py` + `markov_head.sample_block_tokens`），并 **fork `engine_workers.py`** 使其实例化我们的类 | **须 fork** |
| **IP-2** | Actor / Model | 模型由 `engine/fsdp/transformer_impl.py::_build_module`（~230）→`_build_model_optimizer`（543）经 `get_hf_auto_model_class(hf_config)` + `AutoModel*.from_pretrained` 构建（`utils/model.py:686`）；`forward_step`（~1080-1200）取 `output.logits` 形状 `(B,T,V)`（或 rmpad 下 `(total_nnz,V)`），再 `logprobs_from_logits`。**不支持任意 `nn.Module`/自定义 forward 签名**；`models/registry.py` 的 `ModelRegistry` 仅 Megatron 且限 `...ForCausalLM` | fork FSDP engine 的 `_build_module`（改为构建 `Qwen3DSparkModel`）与 `forward_step`/`forward_backward_batch`（草稿无标准 `.logits`、context 来自缓存 hidden） | **须 fork（最深）** |
| **IP-3** | 数据流 | 标准 prompt parquet → tokenizer → DataProto | 把 `CacheDataset`（`prepare_target_cache.py` 产物）桥接为 DataProto，携带 `input_ids/loss_mask/target_hidden_states/target_last_hidden_states` | 中：自定义 dataset + 触及 batch 组装 |
| **IP-4** | Teacher 打分 | 原生 teacher = vLLM/SGLang `prompt_logprobs`（见 A.1，无法承载块对角 forward）；reward_model（`experimental/reward_loop/`）亦为服务化推理，`compute_rm_score` 仅在**末 token**写标量 | 把 target 作为 **in-worker torch 模块**：fork ref-role（`ActorRolloutRefWorker.compute_ref_log_prob`，`engine_workers.py:637`，是「训练 worker 内做逐 token 打分 forward」最近的锚点）或另写 worker/role，做块对角打分 | **须 fork** |

**装饰器可注册（低成本，无需 fork）的部分**：
- 自定义 **advantage estimator** → `@register_adv_est("...")`（0.8.0 **无 `token_reward_direct`**，须自己注册；但 PG-OPD 路径其实**绕过 advantage**，直接把 `−distill_loss.detach()` 喂给 policy loss）。0.8.0 现有：`gae/grpo/grpo_vectorized/grpo_passk/gdpo/reinforce_plus_plus(_baseline)/remax/rloo(_vectorized)/opo/gpg/optimal_token_baseline/tir_optimal_token_baseline`。
- 自定义 **policy loss** → `@register_policy_loss`（**但所有 policy loss 只吃 2D `(B, resp_len)`；0.8.0 policy loss 无 3D top-k 支持**——top-k 分布信号在 distillation logits-processor 的 `compute_topk_loss` 里单独处理，不在 policy loss）。0.8.0 现有：`vanilla`（含 dual-clip）/`gspo/sapo/dppo_tv/dppo_kl/clip_cov/kl_cov` 等。
- 自定义 **distillation loss** → `@register_distillation_loss`（可复用其 `forward_kl_topk`/`k1` 等 KL 库）。
- 自定义 **reward function/manager** → `custom_reward_function.path` / `@register`（可跑任意 torch，但期望标量/末 token；要输出逐 token 张量须覆写 `assemble_rm_scores` 或上游自写 `rm_scores`）。

## A.3 verl 0.8.0 真正「白拿」的部分

即使 IP-1/2/4 要 fork，以下**基础设施**仍可复用：
- Ray single-controller 编排、`WorkerGroup`、`@register` dispatch；
- **FSDP/FSDP2 + Megatron + veOmni/torchtitan** 多引擎后端（`workers/engine/`）；
- checkpoint 引擎、动态批、序列/上下文并行、train↔rollout 权重同步；
- 日志（console/wandb/tensorboard/mlflow）；
- OPD **KL loss 库**（`forward_kl_topk`/`k1`/`k3` 等）与 PG-vs-监督两种范式的成熟实现——数学层面可直接借用。

## A.4 工程量与风险（诚实评估）

DSpark 同时踩中**三处非标准**（非 vLLM rollout、非 CausalLM actor、块对角 teacher forward），而这三处恰恰都是 verl 0.8.0 **硬编码到 HF-CausalLM + vLLM/SGLang + 扁平序列**的「硬缝」，**都落在 fork 而非 plug-in 上**。需要 fork 的清单：
1. `workers/engine_workers.py`（worker 接线 / rollout 选择）；
2. `workers/engine/fsdp/transformer_impl.py`（`_build_module` 模型构建 + `forward_step` 前向/取 logits）——**最深改动**；
3. 一个 `BaseRollout` 子类（块并行采样）；
4. teacher/ref 打分路径（in-worker torch target）；
5. 因 rollout 输出是「anchor×block」结构而非扁平序列，还会触及 `ray_trainer.py` 的 batch 组装、`response_mask`、advantage/loss 管线（0.8.0 中 2D `[B, resp_len]` 假设**无处不在**）。

**推荐落地形态（若最终仍选 0.8.0）**：遵循 verl `recipe/` 的模式——建一个**自包含 recipe 目录**，pin 住 v0.8.0 commit，通过**子类化**上述 worker/engine 实现 fork（参考 `recipe/specRL/histoSpec`、`recipe/gkd/megatron` 的子类化范式，但须 re-target 到 0.8.0 tag，二者当前 pin 的是 pre-0.8 commit）。

> **可行性判断：⚠️ 中偏重（可行，但集成成本显著高于初判）**。无原理阻塞，verl 提供的基础设施与 OPD 数学库价值确凿；但「白拿 advantage/policy-loss/reward worker」的初步设想在 0.8.0 **不成立**——三处非标准点均需 fork engine/worker，且 2D 序列契约与 DSpark 的 block 结构冲突需在多处适配。

## A.5 与 verl 0.7.0 的对比要点

关键反差：**0.8.0 移除/重构掉的、恰是 DSpark-OPD 最需要的现成件**——
- 0.7.0 fork 已有 `token_reward_direct` advantage 与**支持 3D `(B,T,K)` top-k 的 `compute_policy_loss_vanilla`**；0.8.0 两者皆无（policy loss 仅 2D）。
- 0.7.0 的 teacher 走 `RewardModelWorker` 的**普通 torch forward**（`self.reward_module(...)`），可改造为块对角打分并输出逐 token 3D 张量；0.8.0 的 teacher 是 vLLM/SGLang 服务化 `prompt_logprobs`，无法承载自定义 forward。
- 0.7.0 的 rollout/actor/reward 扩展点更接近「单方法覆写 / if-elif 分支」；0.8.0 是新 engine 抽象，需 fork 更深。

因此主文档转向 **verl 0.7.0** 作为首选承载框架（详见 `DSpark-OPD.md` §2.6）。

## A.6 未验证项

- 未能取到 verl 0.7.0 官方 release notes 原文以引用「improved validation and OPD support」的确切措辞——OPD 在 v0.8.0 tag 为原生已由源码确认，但该 release-note 措辞未独立核实。
- `HFRollout`/`NaiveRollout` 在 v0.8.0 存在且被导出，但未找到把它们接线进重构后 `ActorRolloutRefWorker` 的配置路径——按「需自行实例化的 legacy helper」对待。
