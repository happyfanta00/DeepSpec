# DSpark-OPD 优化 #4 设计：worker 侧直读 cache（消除 driver dispatch 单点管道）⚠️ 已实现，暴露新瓶颈

> 状态：⚠️ **已实现并实测**——dispatch 瓶颈确实消除（前段 step 达预期 ~7-9s），但**暴露了新的
> cache 冷读 IO 瓶颈**（step 随访问推进 6.8s→16.8s 单调上升），需配合 §9 的读预取/overlap 才达标。
> 前置优化 #3（repeat-in-worker，已落地 3.2x）见 `fused-step-design.md` §优化#3。

## 0. 实测结论（三方对比，8卡 BATCH=64）

| 方案 | mean | median | min–max | 备注 |
|---|---|---|---|---|
| baseline（driver-repeat） | 56.6s | 51.6s | 21–100s | driver 序列化 3.56GB |
| #3 repeat-in-worker | 17.8s | 15.6s | 10–30s | driver 序列化 0.89GB |
| **#4 cache-read** | **12.5s** | 12.9s | **6.8**–16.8s | dispatch≈0，但读 IO 上critical path |

**#4 呈单调上升**：前段 step2-7 mean **8.9s**（step3 低至 6.8s = 纯 compute，dispatch 已消除，
**架构改对**）；后段 step10-15 mean **15.3s**。差值 ~8.5s 全是 **cache 冷读 IO**。

**根因（受控读基准坐实）**：单进程顺序读 64 样本，耗时随 index 推进 471ms→~1500-2000ms；重复读已
缓存的 idx[0:64] 仅 ~200ms。cache 19TB（608 shard ×68GB，md0 RAID xfs 本地盘，**非** EFS 网络盘）
远大于 RAM page cache，顺序走 index 每 batch 命中冷 shard 区，冷读 page fault 主导。**旧路径为何
无此问题**：driver `dataloader_num_workers=8`（`StatefulDataLoader`，`ray_trainer.py:391`）**预取下一
batch**，读 IO 与当前 step 的 dispatch+compute **overlap**；#4 把读挪进 worker `train_step` 关键路径
且**无预取/overlap** → 冷读串行暴露。**这不是 #4 方向错，是缺了读预取（见 §9）。**

## 1. 动机（实测，非反推）

优化 #3 把 dispatch 量从 3.56GB 降到 0.89GB（去掉 rollout.n 的 4× 冗余副本），step 均值
56.6s → 17.8s（3.2x）。但**数据强制过 driver 单点管道**这个架构病根仍在。py-spy 高频采样
（8卡 BATCH=64，60 次 @0.5s，稳态 step 均值 21.4s）实测新路径 step 拆解：

| 阶段 | 占比 | 绝对秒 | 说明 |
|---|---|---|---|
| `parallel_put`（driver dispatch 序列化） | 60% | **~12.8s** | 单核 pickle 0.89GB hidden，方案 #4 消除 |
| `get_objects`（等 worker compute） | 40% | ~8.6s | rollout+teacher+update 真实计算 |

driver 单核 pickle 有效速率实测 ≈ **69 MB/s**（GIL 下 `parallel_put` 的 16 线程池实际串行，
`top` 观测 driver 100% 单核 `R`、栈在 `__getstate__`）。**即使去掉了冗余副本，剩下的 0.89GB
仍要串行序列化 ~13s**——这就是 SFT 完全没有、而 OPD 独有的开销。

> 注：反推曾估 dispatch ≈12.9s，实测 12.8s，吻合；但反推把 compute 估成 4.9s，实测 8.6s
> （反推用 A/B 均值差、含 step 方差，实测采样更准）。**方案 #4 理论上限：21.4s → ~8.6s，
> 再 ~2.5x**（不是反推的 3.6x；以实测为准）。

## 2. 为什么 SFT 没有这个问题（架构对比）

用户观察正确：SFT 也用 cache 的 target hidden，却毫无性能问题。根因是**两种训练框架的数据
路径架构截然不同**：

| | SFT（`deepspec/trainer/base_trainer.py`，torchrun + FSDP） | OPD（verl + Ray，single-controller） |
|---|---|---|
| 进程模型 | `torchrun` 起 N 个**对等 rank**，无中心节点 | 1 个 driver + N 个 Ray worker actor |
| 数据加载 | **每 rank 自己的 DataLoader** + `DistributedSampler`（`base_trainer.py:289-304`），`__getitem__` 直接从磁盘读**自己那份**分片，`num_workers` 子进程并行预取 + `pin_memory` | driver 单个 DataLoader 读**整个 batch**，再 `parallel_put`（pickle → object store）dispatch 给 worker |
| hidden 流经 | N 条读盘管道并行，**不经序列化** | **全堵在 driver 一根管道**：磁盘 IO（单点）+ pickle（GIL 单核 69MB/s） |

**结论**：不是数据本身大，是 verl 的 single-controller 架构把 per-sample 巨型 side-tensor
（`target_hidden_states [B,T,12800]`）挤进了一个单点串行通道。SFT 的 DDP-style「每 rank 各读
各的」天然没有这个瓶颈。

## 3. 设计：dispatch 只传 index，worker 从 cache 读

> **⚠️ §3–§6 描述的是 #4（`cache` 模式）的原始设计（index-only + worker 读 cache hidden）。
> 实测 #4 暴露 cache 冷读瓶颈（§0），已被 #5（`recompute`，见下方「优化 #5」）取代为默认。
> `cache` 模式保留为非默认选项；下文 index-only 叙述现仅适用于 `mode=="cache"`。**

核心思路——**把 OPD 的数据路径改成 SFT 同构**：driver 不再把 hidden 塞进 dispatch，只传
**样本 index**，worker 各自从 `/mnt/scratch` 读自己 DP 分片的 hidden。

### 3.1 关键前提（已确认）

`CacheDataset[index]` 返回 `input_ids / loss_mask / target_hidden_states /
target_last_hidden_states`（`dataset.py:64-80`），`attention_mask`/`position_ids` 是
`__getitem__` 里从 seq_len 派生的。所以 **worker 只要拿到 index 就能重建 train_step 需要的
全部张量**——dispatch 可以只传一个 `[B]` int64 index 张量（B=64 → **512 字节**，vs 现在 0.89GB）。

### 3.2 数据流对比

| | 优化 #3（当前） | 优化 #4（本设计） |
|---|---|---|
| driver dataloader 产出 | 完整 batch（含 hidden） | **只产 sample index**（`__getitem__` 只返回 index，或 driver 侧从 dataloader 取 index） |
| dispatch payload | 0.89GB hidden + 小张量 | **[B] index 张量（~512B）** |
| pickle 序列化 | 单核 13s | **~0s（几百字节）** |
| hidden 加载 | driver 单点读 + dequant | **8 worker 并行**从 cache 读各自分片 + fp8→bf16 dequant（8× 并行） |
| collate（右填充） | driver 侧填到 batch-max-T | **worker 侧**各填到 per-rank-max-T（可能更小 → 更快） |

### 3.3 DP 分片对齐（正确性核心，与 #3 同理可证）

- driver dispatch 用 `.chunk(dp_size)` **连续切分** index 张量（`decorator.py:78` /
  `protocol.py:887`）→ rank r 拿到 index[r·(B/dp):(r+1)·(B/dp)]。
- worker rank r 用这些 index 从 cache 读 → 与「driver 读完整 batch 再 chunk 给 rank r」拿到
  **完全相同的样本**（cache 是确定性的：同 index → 同张量）。
- rollout.n 的 repeat（#3 已在 worker 侧）：可以对 **index** 做 `repeat_interleave(n)`（更省，
  index 复制几乎免费），或对读出的张量做——二者等价，推荐对 index 做（读盘次数不变，n 份共享
  同一次读）。**⚠️ 注意**：若对 index repeat 后读盘，n 份是同一样本 → 读 1 次即可，用
  `unique index → 读 → repeat 张量`，避免 8×n 次冗余读盘。

### 3.4 worker cache handle 生命周期

- `DSparkActorRolloutRefWorker.init_model` 末尾（`_build_teacher` 之后）构建
  `self._cache = CacheDataset(cache_dir=oc["dspark_target_cache_path"])`（从 override_config 拿
  路径，与 dataset.py 同源）。
- `CacheDataset` 是**只读、mmap/lazy** 的（确认：`deepspec/data/target_cache_dataset.py`），
  8 个 worker 各持一个 handle 读同一目录 = 并发只读，无冲突。
- fp8→bf16 dequant 在 `__getitem__` 里（现 driver 单点，移到 worker = 8× 并行）。

### 3.5 collate 移到 worker

- 现 `dspark_collate_fn`（driver 侧）把变长样本右填充到 **batch 全局 max-T**，chunk 后每 rank
  拿 `[B/dp, max_T, ...]`。
- 方案 #4：worker rank r 读自己 B/dp 个样本，调 `dspark_collate_fn` 填到 **per-rank max-T**
  （≤ 全局 max-T）。**数值等价**：padding 由 loss_mask 屏蔽，各 rank forward 本就独立；per-rank
  T 更小反而**可能更快**（少算 padding）。这正是 SFT 每 rank 独立 collate 的行为。

## 4. 改动清单

1. **`dataset.py`**：新增「index-only」模式——`__getitem__` 只返回 `{"sample_index": index}`
   （或 driver 侧 dataloader 用一个轻量 IndexDataset）。保留原 CacheDataset 供 worker 用。
2. **`trainer._prepare_batch`**：只把 index 包成 DataProto（`batch={"sample_index":[B] long}`），
   不再读 hidden。repeat 仍延后到 worker（对 index）。
3. **`worker.init_model`**：构建 `self._cache`。
4. **`worker.train_step`**：开头把收到的 index → unique → `self._cache` 读张量 →
   `dspark_collate_fn` collate → repeat_interleave(n) → 进入现有 Phase1-3（其余不变）。
5. **config**：确保 `dspark_target_cache_path` 在 override_config（worker 可见）。

## 5. 风险与开放问题

- **⚠️ 跨节点 cache 可达性**：单节点 8 卡同机，worker 都能挂 `/mnt/scratch`（与读 draft/teacher
  权重同源，已验证可达）。**多节点时** worker 未必都能访问同一 cache 路径——需共享 FS（EFS/Lustre
  已是共享，应 OK，但要显式确认）。SFT 也依赖此前提（DistributedSampler + 共享 cache），故不是新约束。
- **读盘带宽**：8 worker 并发读 `/mnt/scratch`（EFS/Lustre）。SFT 已这么跑且无瓶颈，但 SFT 的
  batch/卡可能不同；需实测 8× 并发读的聚合带宽是否 > 现单点 IO+序列化（几乎必然，因省了序列化）。
- **dataloader 语义**：driver 仍需一个 dataloader 驱动 sampler/shuffle/epoch，只是它现在产 index。
  `StatelessResumableDistributedSampler`（SFT 用的）vs verl dataloader——需确认 index 的
  shuffle/顺序与现行为一致（否则改变训练数据顺序，虽不影响正确性但影响可复现）。
- **metrics collect 不变**：train_step 仍只回标量 metrics，collect 路径零改动。
- **balance_batch**：per-rank 变长 T 下，token 数跨 rank 更不均（现已有此问题，见 grad-sync
  memory ②）；#4 不改善也不恶化，仍是 open 项。

## 6. smoke 策略（正确性不回归）

- **`worker_cache_read_smoke.py`（纯 dataflow，无 GPU，仿 `repeat_in_worker_smoke.py`）**：
  - A) 「driver 读完整 batch 再 chunk」vs「dispatch index 再 worker 按 index 读 cache」→ 每 rank
    拿到的样本张量 **bit-identical**（cache 确定性 + chunk 连续性）。
  - B) per-rank collate（填到 per-rank max-T）vs 全局 collate 后 chunk → loss_mask 有效位对齐、
    有效 token 的张量值相等（padding 差异被 mask，不影响）。
  - C) repeat：unique-index 读 + repeat 张量 == 现 worker-repeat 张量（bit-identical）。
  - D) 跨 (B,dp,n) sweep + dp=1 + `B%dp≠0` 断言。
- **E2E A/B 测速**：`DSPARK_HIDDEN_MODE` env 开关（默认 `recompute` #5；`=cache` 走 #4；`=dispatch`
  或 legacy `DSPARK_CACHE_READ_MODE=0` 走 #3 传 hidden），8卡
  BATCH=64 各 10 step，对比 step wall。预期 21.4s → ~9-11s（dispatch 段 ~13s → ~0）。

## 7. 决策记录

- 用户 2026-07-09 观察：SFT 用同样 cache hidden 无性能问题，本质是 verl 数据全过 driver 单管道
  （IO + 序列化），提议 worker 自读 cache。**方向确认**：先实测（本文 §1 已完成，dispatch=12.8s/
  step=60%）再写设计（本文）→ 待评估回归风险后决定实现。
- 与 #3 的关系：#3（repeat-in-worker）是**必要的独立优化**（去 4× 冗余），已落地；#4 在 #3 基础上
  进一步消除**剩余单点 dispatch**。二者正交，#4 保留 #3 的 worker-repeat 逻辑（改成对 index repeat）。
- 用户 2026-07-09（续）：#4 实现后实测暴露 cache 冷读 IO（§0），要求"更严谨性能分析"→ §8/§9。

## 8. 严谨性能分析（已完成，定位 cache 冷读 IO）

分三层测量，逐层排除，锁定瓶颈：
1. **端到端 step wall（trainer 打印）**：#4 = 6.8s→16.8s 单调升，非稳态 → 有累积开销。
2. **受控读基准（单进程，无 GPU/训练，`CacheDataset` 直读）**：顺序读 64 样本 471ms→2058ms，
   重复读热数据仅 ~200ms → **累积开销 = 磁盘冷读**，非内存泄漏/句柄泄漏（`max_open_shards=4`
   LRU 正常 evict，`target_cache_dataset.py:760`）。
3. **文件系统确认**：cache 在 `/dev/md0`（xfs on RAID 本地盘，**非** EFS 网络盘），19TB≫RAM →
   page cache 无法容纳，顺序遍历必冷读。
4. **架构对比**：旧路径 driver `dataloader_num_workers=8` 预取 overlap 掉了读 IO；#4 读在
   `train_step` 关键路径且无预取 → 暴露。

**结论**：#4 的架构改动（消除 dispatch）**正确且必要**（前段已达 ~7-9s 预期）；瓶颈转移到**读 IO
未 overlap**。需 §9 补预取。

## 9. 修复方向：cache 读与 compute overlap（待实现/讨论）

核心：把 worker 侧 cache 读**移出 train_step 关键路径**，与上一 step 的 compute overlap（复刻旧
路径 driver dataloader 预取，但在 worker 侧）。候选：

- **(a) worker 侧后台预取线程**：worker 收到 step t 的 index 后，起后台线程读 step t 的 cache，
  同时主线程跑 compute；下一 step 复用。但 train_step 是**同步 RPC**（driver 逐 step 调），worker
  拿不到 t+1 的 index → 需 driver 提前多 dispatch 一个 index batch（pipeline depth=1）。改动中等。
- **(b) driver 侧仍预取张量，但走 SFT 式每-rank dataloader**：放弃 verl dispatch，worker 各自起
  `DataLoader(DSparkCacheDataset分片, num_workers>0, prefetch)` —— 最贴 SFT，但要 worker 侧完全接管
  数据加载（绕过 verl dispatch，回归风险高）。
- **(c) 降低冷读量**：hidden fp8 存储（现 cache 已是 float8_e4m3fn 落盘？确认），或只读实际用到的
  layer/token 子集；治标。
- **(d) 预热/常驻**：训练前把工作集 shard 预读进 page cache（若 n_samples×11MB < RAM 可行；20000
  样本 ×11MB=220GB，超 RAM，不可行；小 n_samples 可行）。
**受控读基准结论（单进程，已测；训练是 8-worker 并发、每 worker 读 B/dp=8 样本，模式略不同但趋势同）**：
- 单样本重复读：冷 168ms → 热 2.5ms（**page cache 生效**，2TB RAM、buff/cache 1.7TB 充裕）。
- 同批 idx[0:64] 连读：388→200ms 转热，**热读下限 ~200ms/64样本**（`.copy()`+tensor 构造，可忽略）。
- **顺序遍历（每 batch 走新 index 区间）**：471→2058ms 冷读；**跨 shard**（idx 每隔 2000）达 **2029ms**。
- **∴ 瓶颈 = 大数据集顺序遍历时每 batch 命中冷 shard offset 的首次磁盘读**，热数据无问题。
- **SFT 无此问题**：SFT 每 epoch 复用同样本、driver `num_workers=8` 预取 overlap；#4 读在 worker
  关键路径且无预取。n_samples 小（工作集 < RAM）时 OPD 也会转热，但 n_samples=20000（220GB）
  会持续冷读。

**下一步（待用户定）**：推荐 **(a) worker 侧预取 + pipeline depth=1**（driver 提前多 dispatch 一个
index batch，worker 后台线程预读下一 step 的 cache，与当前 compute overlap）——复刻旧路径 driver
预取的收益、但保留 #4 的零 dispatch。预期把 step 拉回前段的 ~7-9s（compute-bound）。次选 (d) 小
n_samples 预热（仅小规模实验可行）。

---

# 优化 #5：worker 内 teacher 重算 hidden（✅ 已实现，替代读 cache hidden）

> 用户 2026-07-10 定的方向。**思路**：既然 teacher（Qwen3-4B）已常驻 worker（Phase 2 打分用），
> 就不再读/传 `target_hidden_states`，改为 worker 内用 teacher 对 `input_ids` 做**一次 prefill
> forward + decoder-layer hooks** 重算它（`target_layer_ids=[1,9,17,25,33]` 拼接，与 cache 生成
> `prepare_target_cache.run_target_forward_with_hooks` 完全同构）。**用一次 ~0.1s GPU forward 换掉
> cache 冷读 + 序列化。**

## 5.1 三档模式（`DSPARK_HIDDEN_MODE`）

统一到一个 env（取代 `DSPARK_CACHE_READ_MODE`；后者 =0 仍兼容映射到 `dispatch`）：
- **`recompute`（默认，#5）——标准 verl 数据流**：driver dataloader 读 **token**（`read_tokens_only`，
  input_ids/loss_mask/attention_mask/position_ids，~KB/样本，`num_workers` 预取 overlap）并**标准
  dispatch**；worker 用常驻 teacher **重算** hidden。**不读/不 dispatch 大 hidden。**（用户 2026-07-10
  订正：recompute 下应回归 driver 分发完整训练数据的标准做法——token 由 driver 读+分发，worker 只
  重算 hidden；不再走 index-only + worker 按 index 读 token 的非标准路径。）
- **`cache`（#4）**：dispatch 只传 index；worker 读**完整** cache 样本（含 hidden，付冷读代价、无预取
  overlap）。仅此模式是 index-only。
- **`dispatch`（#3）**：driver 读全张量（token+hidden）+ 序列化 dispatch（legacy）。

## 5.2 关键发现：recompute 是 teacher 真实 hidden，cache 是 fp8 有损版

`recompute_hidden_smoke.py` 实测：recompute（bf16）vs cache——**方向一致（cos 0.998，layer1 达
0.9997），但深层范数差 1.2–3.1×**（layer9 ratio 3.07）。根因：cache 存 fp8（float8_e4m3fn），
`sanitize_fp8_hidden_states` 把超 ±448 的 **outlier 特征维饱和截断**（Qwen3 深层 hidden 幅度上千）；
recompute 保留真实幅度。**这不是 bug——recompute 更精确**。
- **用户决策（2026-07-10）：用未截断真实 hidden（不加 fp8 clamp）。**
- **隐藏正收益**：推理/eval 的 target hidden 走**实时 teacher forward**（`output_hidden_states=
  True`，bf16，`base_evaluator.py:217`），**从不用 fp8 cache**。故 #5 recompute 恰好 == 推理路径 →
  **消除了原训练(fp8 cache)/推理(bf16 实时)的分布 gap**，而非引入。
- batch-invariance：padded batch vs 单样本 unpadded，real-token relmax=0.0006（纯 forward 等价，
  与 S2/S3 同量级）。

## 5.3 预期性能（实测 forward 开销 + E2E 待测）

单卡基准：teacher 对 per-rank 8 unique 序列（T~435）重算 hidden ~79ms、T~543 ~100ms（只对 unique
重算、repeat 前，n=4 副本共享）。预期 step ≈ #4 前段纯 compute（~6.8s）+ 重算（~0.1s）≈ **~7s**，
**无磁盘冷读、无大张量 dispatch、无 IO 抖动/暖机问题**。四方对比见 memory。

## 5.4 改动清单（已实现）

1. `teacher_scoring.recompute_target_hidden_states`（新，hook forward）+ `_teacher_backbone`。
2. `target_cache_dataset.CacheDataset.read_tokens_only`（新，只读 input_ids/loss_mask，避冷读 hidden；
   与 `__getitem__` 共享 `_open_sample`/`_read_tokens` 私有 helper）。
3. `dataset.py`：`DSPARK_HIDDEN_MODE` 三档；`__getitem__` 仅 `mode=="cache"` 返回 `{sample_index}`
   （index-only），`"recompute"` 返回 `adapt_tokens_only(read_tokens_only(idx))`（仅 token 张量，标准
   dispatch），`"dispatch"` 返回 `adapt_cache_record(cache[idx])`（含 hidden）。`adapt_cache_record`
   复用 `adapt_tokens_only` 再挂 hidden。
4. `worker.init_model`：`recompute` 模式建 `self._target_layer_ids`（从 draft 读，`self._cache=None`）；
   `cache` 模式建 `self._cache` + `self._target_layer_ids`（从 cache 读）；均设 `self._hidden_mode`。
5. `worker.train_step`：materialize 块按 `_mode` 分三支；recompute 支只对 unique 序列跑 teacher 重算再 repeat。
6. config `override_config.dspark_target_cache_path`（worker 可见）。
7. smoke：`recompute_hidden_smoke.py`（数值/方向/batch-invariance）；`s5_recompute_stage_dump.py`（S1-S4 各阶段）。

## 5.5 待优化点（非阻塞）

- **teacher 双 all-gather**：recompute forward + Phase2 打分各 all-gather 一次 FULL_SHARD teacher
  参数。可合并（打分本就 prefill context 0..anchor，理论上能顺带抓 hidden），省一次 gather（~40ms）。
  当前分开写，清晰优先；E2E 若显示 gather 占比大再合并。
