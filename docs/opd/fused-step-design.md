# DSpark-OPD 融合训练步设计（#2：消除 Ray object-store 大张量往返）✅ 已实现

> 目的：把 rollout → teacher → update 三次独立 worker RPC 合并为**一次 dispatch**，中间产物
> 留在 worker 显存，只回传标量 metrics，消除每 step ~800MB+ 的 Ray object-store 序列化/往返
> （py-spy 实测瓶颈在 `parallel_put`，见 memory [[dspark-opd-multigpu-grad-sync-bug]] / 本次诊断）。
>
> **实现状态（已落地）**：走**方案(2)——teacher 并入 actor worker**（`_build_teacher`，取消独立
> RewardModel role）。融合方法 = `DSparkActorRolloutRefWorker.train_step`（`@register(mesh_name=
> actor)`）：Phase1 rollout(`dspark_block_rollout`,no_grad,unwrapped submethods) → Phase2 teacher
> (`score_blocks_flat` on `self.teacher_module`,no_grad,末尾 reshard) → Phase3 update(micro 循环,
> `module(...)` grad,`_optimizer_step`)。gate 机制已删除(`gate.py`/`stop_after`/`STOP_AFTER`/`dump`)。
> `trainer.fit` 简化为多 step 循环调 `train_step`。正确性:`fused_step_smoke.py`(per-micro 切分 ==
> 整批梯度 `max|Δ|=6.3e-3`、determinism、冻结无梯度) + s3/s4/grad_sync smoke(底层纯函数)全过。
> sibling 访问方案(甲)未采用(需改 verl core 切 fused colocated worker,回归风险高)。

## 后续优化 #3：rollout.n repeat 挪到 worker 侧（✅ 已实现 + 实测 3.2x 加速）

> **⚠️ 后续演进（见 `worker-side-cache-read-design.md`）**：#3 之后又做了 #4（worker 读 cache）和
> **#5（worker 内 teacher 重算 hidden，现为默认 `DSPARK_HIDDEN_MODE=recompute`）**。**默认模式下
> driver 只 dispatch token（~KB），`target_hidden_states` 由 worker 重算**——本节下面「driver 传
> batch 一次(400MB)」「~400MB 上传」的数据流表述仅适用于 legacy `dispatch` 模式（#3）。融合本身
> （三阶段一次 RPC、只回标量）与 repeat-in-worker 逻辑不变。step 均值 56.6s（baseline）→ 17.8s(#3)
> → 12.5s(#4) → **8.5s(#5，实测 8 卡 200-step)**。

> **诊断（融合后仍有长空窗，py-spy + nvidia-smi dmon + top 实测）**：融合消除了 update 阶段的
> **重复**传输,但每 step 仍要 dispatch 一次输入 batch。而 `trainer._prepare_batch` 沿用 verl 标准
> 做法在 **driver 侧** `batch.repeat(rollout.n=4)` **再** dispatch —— 把每 prompt 的
> `target_hidden_states [B,T,12800] bf16` 物理复制 4 份(64→256 序列 ≈ **3.56GB**,其中 **75%
> (2.67GB) 是 bit-identical 冗余副本**:同 prompt 的 n 个 rollout 只是各自采不同 anchor,hidden 完全
> 一样)。`parallel_put`(`ray_utils.py:73`)再单核 pickle 这 3.56GB(driver 100% CPU `R`、`__getstate__`,
> **GIL 下 16 线程池实际串行**)→ 实测每 step 空窗 **21–100s、均值 56.6s**(这就是"超过 1 分钟空窗"来源)。
> object store 186GB 只用 30GB、`/dev/shm` 843GB 空闲 → **非 spill/IO,纯 CPU 序列化**。
>
> **修法**:rollout.n 的 `repeat_interleave` 从 driver 挪进 `train_step`(worker 侧,`data.to(device)`
> 后、Phase1 前)。driver 只 dispatch **64 unique** prompt(~0.89GB),序列化量 **−75%**;worker 收到
> 后在 GPU 上 repeat。**数值 bit-identical**:verl dispatch 用 `.chunk(dp_size)` 连续切分
> (`decorator.py:78`/`protocol.py:887`),interleave 组(size=n)在每 rank 的连续 chunk 内整除平铺,故
> "repeat→chunk"(旧)与"chunk→repeat"(新)每 rank 逐字节相同。**前提 `train_batch_size % dp_world == 0`**
> (旧只需 `×n % world==0`,新更严;64%8=0 满足)。
>
> **实测 A/B(8卡 BATCH=64,同进程 env 开关 `DSPARK_REPEAT_ON_DRIVER` 切换,10 step)**:
> | | mean | median | min–max |
> |---|---|---|---|
> | driver-repeat(旧) | 56.6s | 51.6s | 21–100s |
> | worker-repeat(新) | **17.8s** | **15.6s** | 10–30s |
> | **加速** | **3.2x** | 3.3x | — |
>
> loss 同分布(step1-3: 旧 0.50/0.24/0.51 vs 新 0.47/0.28/0.55,仅随机采样差异),两次 rc=0。
> **正确性验证**:`scripts/opd/repeat_in_worker_smoke.py`(纯 dataflow、无 GPU):real config +
> (B,dp,n) sweep + dp=1 + `B%dp≠0` 触发 chunk 断言 + 全局拼接重建 = 旧张量,全 bit-identical。
> **A/B env 开关 `DSPARK_REPEAT_ON_DRIVER` 保留**(默认 worker-repeat;=1 复原旧路径,仅供再测)。

## 诊断回顾（已用 py-spy + 源码确认）

- 8 卡全 0% util 期间：fit-actor 阻塞在 `parallel_put`（dispatch 序列化 DataProto 到 object store）；8 个 WorkerDict 全 idle in `main_loop`（等 RPC）。
- 每 step 大张量往返：`target_hidden_states [B,T,L*H]` bf16 ≈ **400MB**，rollout dispatch 传一次 + update dispatch 又传一次；加上 rollout 输出（tokens/top-k）、teacher 输出（logp_target）也 collect 回 driver 再 re-dispatch。
- **根因**：三次 RPC 各自 dispatch(大张量上传) + collect(结果回传)。

## 关键架构事实（源码确认）

1. **actor 与 teacher 同进程**：config 未开 `reward_model.enable_resource_pool` → `Role.RewardModel` 映射到 `global_pool`（与 actor 同池）→ `create_colocated_worker_cls` 把两个 role 合进**一个 `WorkerDict` 进程**（每卡 ~52GB = draft+optimizer+teacher）。所以 rollout/teacher/update 本就在同一进程同一 GPU，中间张量**无需跨进程**。
2. **dispatch chunk 确定性**：`dp_rank_mapping[i]` 固定，同一 batch 每次切给同一 rank。
3. **legacy colocated worker**：`worker_dict["ActorRollout"]` 与 `worker_dict["RewardModel"]` 是同进程两个实例，但**默认不互相持有引用**（`WorkerDict.__init__` 不注入 sibling）。

## 设计：单个融合 actor 方法 `train_step`

**放在 `DSparkActorRolloutRefWorker`**（actor worker），`@register(mesh_name="actor")`。一次 dispatch
收到 batch，内部顺序跑完整个 step，只回标量：

```
@register(mesh_name="actor")
def train_step(self, data):                      # data: input_ids/loss_mask/target_hidden_states (一次上传)
    b = data.batch.to(device)
    # 1) rollout (no_grad) —— 复用 dspark_block_rollout,产物留显存(不回 driver)
    roll = dspark_block_rollout(self.actor.actor_module, input_ids, loss_mask,
                                target_hidden_states, num_anchors, temperature, top_k)
    # 2) teacher 打分 (no_grad) —— 同进程直接调 teacher,产物留显存
    T_on_S = score_blocks_flat(self._teacher_module, input_ids, roll.tokens, ...)
    # 3) update (带梯度) —— 复用现有 update_dspark_opd 主体
    metrics = _opd_update(self.actor.actor_module, roll, T_on_S, ...)
    return DataProto(metrics)                    # 只回标量 [1] 张量
```

**跨 role 拿 teacher（`self._teacher_module`）**：legacy colocated worker 不注入 sibling 引用，
故需显式 wiring。**方案（择一，实现时定）**：
- **(A) 复用现有分离 worker，但融合在 trainer 侧仍是 1 次 dispatch**：❌ 不行——trainer 调 `actor_rollout_wg.train_step` 只到 actor 进程,拿不到 teacher。
- **(B) 融合方法内构建/持有 teacher**：让 `DSparkActorRolloutRefWorker` 在 `init_model` 后**也持有 target 模型**（额外建一个 teacher FSDP，或复用 reward worker 的构建逻辑）。代价：actor 进程显存已含 teacher（同进程），但要避免重复构建。
- **(C) ★推荐：通过 WorkerDict 的 `worker_dict` 兄弟访问**——在 `init_model` 末尾把 sibling teacher 存到 `self`。因为 bound 方法在 `worker_dict["ActorRollout"]` 上执行，它能否看到 `worker_dict["RewardModel"]`？默认不能。需在 `WorkerDict.__init__` 后注入,或用 Ray actor handle 自引用拿同进程 sibling。**这是本设计最需要在实现时敲定的机制点**——见下「实现前必确认」。

## 去掉 gate 机制

- **动机**：融合后 rollout/teacher 中间产物不回 driver，`STOP_AFTER=rollout/teacher/loss` 的
  "driver 在 phase 间 dump + exit" 无法工作。且 gate 是 S0-S5 开发期脚手架，S1-S5 已全部验证通过，
  S6/正式训练用 `off`，gate 已完成使命。
- **改动**：
  - `trainer.py`：`fit()` 去掉 gated 分支，只保留多 step 循环，每 step 调 `train_step`（一次 dispatch）。
  - `gate.py`：删除或保留为 no-op（`maybe_gate` 不再被调用）。
  - `_run_opd_step`：并入 `train_step`（worker 侧）+ trainer 侧只做"取 batch → repeat(n) → train_step → 打印 metrics → 按 save_freq checkpoint"。
  - `config`：`stop_after` 字段废弃（或保留但忽略）。
  - `run.sh`：`STOP_AFTER` 旋钮废弃（或忽略）。
- **保留的旧路径**：分离的 `generate_sequences`/`compute_rm_score`/`update_dspark_opd` 方法**保留**
  （不删），因为 smoke 仍可能单独调它们验证；只是 trainer 不再逐 phase 调。

## 数据流对比

| | 现状（三次 RPC） | 融合（一次 RPC） |
|---|---|---|
| dispatch 上传 | rollout 传 batch(400MB) + update 再传 batch(400MB) | **train_step 传 batch 一次(400MB)** |
| collect 回传 | rollout 输出(tokens/top-k) + teacher 输出(logp_target) 均回 driver | **只回标量 metrics** |
| re-dispatch | rollout 输出→teacher，teacher 输出→update 再上传 | 无（中间产物留显存） |
| 每 step object-store 流量 | ~800MB+ 大张量多次往返 | **~400MB 上传 + 几个标量** |

## smoke 策略（★正确性不回归）

**核心断言：融合 `train_step` 的 loss/grad == 分离三步（rollout→teacher→update）的 loss/grad。**

- **单卡 smoke（新增 `scripts/opd/fused_step_smoke.py`）**：同一 batch/seed，
  - 路径 A：分离三步（现有 `dspark_block_rollout` + `score_blocks_flat` + update 逻辑）；
  - 路径 B：融合 `train_step` 内部逻辑；
  - 断言：final loss、grad_norm、各可训练参数 grad **`allclose`**（融合只是把三步串起来，数值必须一致）。
- **复用现有 smoke**：s3_smoke（teacher 批处理等价）、s4_smoke（reward/PG/backward/micro）、
  s4_grad_sync_smoke（多卡梯度同步）——这些验证的是**被融合调用的底层逻辑**，仍有效。
- **多卡 E2E（你执行）**：`STOP_AFTER` 去掉后，`bash run.sh` 直接多 step；py-spy 再抓一次确认
  `parallel_put` 空窗消失、util 上升。

## 实现前必确认（唯一未定机制点）

**融合方法如何在 actor worker 内拿到同进程的 teacher module？** 三条候选：
1. actor `init_model` 里额外构建 teacher（`_build_model` 逻辑复用），actor 进程自持 teacher。
   —— 最自包含,但要确保不和 colocated 的 RewardModel worker **重复建 teacher**（浪费显存）。
2. 关掉独立 RewardModel role，teacher 完全并入 actor worker（不再有 `rm_wg`）。
   —— 最干净的融合,但改动 role 注册（task_runner.add_reward_model_worker）。
3. 通过 Ray 自 actor handle 拿同进程 sibling 的 `reward_module`。—— 最省显存但最 tricky。

**倾向 (2)**：既然要融合、且 teacher 只在融合方法里用，不如**取消独立 reward worker role**，让
actor worker 直接持有 teacher（`init_model` 里建 draft + teacher + optimizer）。这样：
- 无重复 teacher、无跨 role 访问难题；
- `train_step` 内 `self.teacher_module` 直接可用；
- 连带简化 resource pool（不再需要 RewardModel role 映射）。
- 代价：`DSparkActorRolloutRefWorker` 承担 teacher 构建 + reshard（把现 `DSparkRewardModelWorker`
  的 `_build_model`/reshard 逻辑搬进来）。

> 待你确认走 (2) 还是 (1)，再进入实现。

---

## 潜在问题调研结论（3 路并行 + 源码/FSDP 实证）

**①b 【实现中发现的真实陷阱，已修】FSDP root lazy-init 顺序**：融合后 rollout(Phase1)在同一
方法内先调**嵌套 FSDP 的 decoder layer 子方法**(`unwrapped._forward_backbone` → `for layer:
layer(...)`)——而 actor FSDP 因 `_no_split_modules=["Qwen3DSparkDecoderLayer"]` 把每层 auto-wrap
成**嵌套 FSDP 实例**。在 root 首次 `forward()` 之前直接调嵌套层,会让每个嵌套层抢先置 `_is_root=
True`,随后 Phase3 的 `module(...)` root `_lazy_init` 触发断言 `Non-root FSDP instance's _is_root
should not have been set yet`。**老的分离路径没踩到**:继承的 `generate_sequences` 用 `rollout_mode()`
→ `actor_module_fsdp.state_dict()` 隐式 prime 了 root。融合 `train_step` 绕过它,故须**在 Phase1 前
显式 prime**:`torch.distributed.fsdp._runtime_utils._lazy_init(module, module)`(幂等,首步后 no-op)。
**验证(三层)**:① 2-rank toy(nested transformer-auto-wrap + submethod-before-root-forward)复现断言;
② 真实模型隔离 2-rank 测试(`FUSED_ORDER_OK`:真 `Qwen3DSparkModel`+真 `dspark_block_rollout` 嵌套子方法
→真 root forward+backward)不 prime 必挂、prime 后 OK;③ **2-GPU 3-step 融合 E2E 实跑 rc=0**
(step1/2/3 loss=0.449/0.771/0.466、grad_norm 有界、ppo_kl=0、双卡落盘)。**下面①的"无风险"结论是在
裸模型上得出的,漏了嵌套 FSDP 这一层——多卡实跑才暴露、现已修复并确认。**

**① FSDP 状态机（no_grad rollout → no_grad teacher → grad update 同方法内）——✅ 无风险（但见 ①b 的 lazy-init 顺序陷阱）。**
- rollout(`dspark_block_rollout`,`@torch.no_grad`,调 submethod,绕过 FSDP.forward)与 teacher(no_grad `module(...)`)都不建图；update 走 `module(...)`(FSDP.forward)带梯度。
- `FSDP._root_pre_forward` **每次 forward 重置状态**(`handle._needs_pre_forward_unshard=True`、`_reset_flat_param_grad_info_if_needed`),故前面的 no_grad 调用**不污染**后面的 grad forward。
- dropout=0(config `attention_dropout=0.0`,无其它 dropout)→ train/eval 模式数值无关。
- `@torch.no_grad` 是函数装饰器,scope 随 rollout 返回结束,update 在其外带梯度运行。

**② 显存——✅ 无风险,delta 仅 MB 级。**
- 现状分离路径:rollout 只回传**小 top-k 张量** `[B,A,blk,K]`(~2-3MB);巨大的 `corrected_logits [B,A,blk,V]`(2.18GB bf16)、`logp_all`(4.36GB fp32)是 rollout 内 no_grad 局部,**返回即释放**,从不回 driver。teacher 的 `[B,Q,V]` logits(4.55GB)、mask(14MB)同理返回即释放。
- 融合后:三个子调用仍是"函数返回只留小张量",大 no_grad 瞬态在各自 `return` 时释放(在 update backward **之前**);峰值 ≈ max(各阶段峰值)+ 常驻基线,**与现状相同**。跨阶段只多存 ~2-3MB 小 top-k。
- 实测 timed run ~52GB/81GB(29GB 余量)→ **无 OOM 风险**。
- **⚠️ 唯一会爆的写法**:若"优化"成把 rollout 的 `corrected_logits [.,V]`(2.18GB)或 teacher 的稠密 `logp [.,V]`(4.36GB)直接传给 update 跳过重算 —— 那会加几 GB、威胁余量。**融合必须保持"update 用 module(...) 重算 draft_logits"**(现状即如此),不传全词表张量。

**③ gate / smoke / 可维护性——⚠️ 改动面广但可控。**
- **smoke 全部安全**:s2/s3/s4/s4_grad_sync **只调纯函数**(`dspark_block_rollout`/`score_blocks_flat`/`loss_bridge.*`)+ `from_pretrained` 直接建模型,**不走 Ray/worker RPC/gate**。融合不打断任何 smoke。
- **修正设计文档一处错误**:上文"保留三个分离方法因 smoke 可能单独调"——**不成立**,smoke 从不调这三个 RPC 方法。故保留分离方法**买不到测试覆盖**,反造成两条同逻辑 live 路径静默漂移 → **应 REPLACE 不 keep-both**(纯函数保留,分离 RPC 方法删)。
- **正确性保障**:一次性 `fused_step_smoke.py` 断言 融合 `train_step` loss/grad `allclose` 分离三步。
- **去 gate 必须一个 commit 同步改**(否则 Hydra "key not in struct" 启动即挂):trainer(去 gated 分支/maybe_gate)、task_runner(去 GateStop/`_PHASES_NEEDING_WORKERS`/init_workers 门控)、gate.py(删/no-op)、run.sh(去 `STOP_AFTER`)、config(去 `stop_after`/`dump`)、+ 4 份文档(DSpark-OPD.md 方法论/阶段表/各 STOP_AFTER 命令、env-setup GateStop、tensor-contract phase 标注)。
- **调试能力**:`_print_rollout_text`/`_print_teacher_signal` 搬进融合 `train_step`,挂 `DSPARK_DEBUG` env/config 开关,替代 gate 的人工核验。

## 总结论

融合**技术上安全**(FSDP 状态无污染、显存中性),**收益大**(消除 timed run 里占 75-80% 的传输,rollout 13-24s + update 400MB 重传)。主要成本是**去 gate 的改动面**(6 代码文件 + 4 文档,须一 commit)。teacher 访问机制走 **(2) 取消独立 role、并入 actor** 最干净(sibling 访问需改 verl core、回归风险高;方案(2)只改 recipe)。
