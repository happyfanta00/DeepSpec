# DSpark 训练调用链路

本文梳理训练 DSpark 模型时 `train.py` 的完整调用链路，以及模型结构、forward 过程、loss 计算分别位于何处。以 `config/dspark/dspark_qwen3_8b.py`（Qwen3-8B）为例。

## 一、启动与调度链路

```
train.py
  └─ torch.multiprocessing.spawn(main, nprocs=每张GPU一个进程)   train.py:45
       └─ main(local_rank)                                       train.py:31
            ├─ parse_args() → 读取 --config 配置                 train.py:20
            ├─ trainer = args.train.trainer_cls(local_rank, args) train.py:36
            ├─ trainer.train()                                   train.py:37
            └─ trainer.clean_up()                                train.py:38
```

`trainer_cls` 由配置文件决定。`config/dspark/dspark_qwen3_8b.py:32` 指定为 `Qwen3DSparkTrainer`，
所以训练 DSpark 时实例化的是 `deepspec/trainer/dspark_trainer.py:14` 的 `Qwen3DSparkTrainer`（继承自 `BaseTrainer`）。

## 二、模型构建链路（在 `BaseTrainer.__init__`）

```
BaseTrainer.__init__                            base_trainer.py:151
  └─ self.build_models()                        base_trainer.py:244
       ├─ AutoTokenizer / AutoConfig.from_pretrained("Qwen/Qwen3-8B")
       ├─ self._build_draft_model(...)          ← 子类实现
       │    └─ Qwen3DSparkTrainer._build_draft_model   dspark_trainer.py:17
       │         ├─ build_qwen3_draft_config(...)      qwen3/config.py:9
       │         └─ Qwen3DSparkModel(draft_config)     qwen3/modeling.py:201  ← 真正的模型
       └─ 加载 target 模型，把它的 embed_tokens / lm_head 拷进 draft 并冻结
            └─ draft_model.initialize_embeddings_and_head(freeze=True)  modeling.py:270
  └─ torch.compile + FSDP 包装                   base_trainer.py:178-181
```

**实际的模型结构在 `deepspec/modeling/dspark/qwen3/modeling.py`：**

- `Qwen3DSparkModel`（modeling.py:201）—— 草稿（draft）模型主体，组成：
  - `embed_tokens`、`lm_head`（从 target 拷贝并冻结，modeling.py:246、modeling.py:270）
  - `layers`：`num_draft_layers=5` 层 `Qwen3DSparkDecoderLayer`（modeling.py:154）
  - `fc`：把 target 多层 hidden 拼接（`len(target_layer_ids) * hidden_size`）投影回 `hidden_size`（modeling.py:240）
  - `hidden_norm` / `norm`、`rotary_emb`
  - `markov_head`（`markov_head.py` 的 `build_markov_head`）
  - `confidence_head`（`AcceptRatePredictor`，common.py:43）
- `Qwen3DSparkAttention`（modeling.py:43）是 DSpark 特有的注意力：把 **target 的 hidden states 当作 context 的 K/V**（`k_ctx/v_ctx`），把噪声 embedding 当作 query / 新位置的 K/V（`k_noise/v_noise`），拼接后做注意力（modeling.py:103-114）。

`Qwen3DSparkModel` 继承 HuggingFace 的 `Qwen3PreTrainedModel`，但层是自定义的，不是直接复用 HF 的 Qwen3 模型。

## 三、训练循环与 forward / loss 链路

```
trainer.train()                                 base_trainer.py:348
  └─ for batch in prefetcher:
       └─ run_batch(batch)                       ← 子类实现
            ├─ self.model(input_ids, target_hidden_states,
            │             loss_mask, target_last_hidden_states)
            │     → Qwen3DSparkModel.forward      modeling.py:388   ← forward 在这里
            └─ compute_dspark_loss(outputs, ...)  loss.py:255       ← loss 在这里
       └─ loss.backward() / optimizer.step()
```

`run_batch` 位于 `deepspec/trainer/dspark_trainer.py:25`。

### forward 过程 — `Qwen3DSparkModel.forward`（modeling.py:388）

关键步骤：

1. `sample_anchor_positions` —— 在序列里采样 `num_anchors=512` 个 anchor 位置（common.py:123）。
2. `create_noise_embed` —— 构造 draft 输入：每个 block 首位放 anchor token，其余位置放 `mask_token_id`（common.py:264）。
3. 构造位置 id 和 **DSpark 专用注意力 mask**（`create_dspark_attention_mask`，用 flex_attention 的 block mask，common.py:78）。
4. `_forward_backbone`（modeling.py:361）—— 把 target hidden 经 `fc + hidden_norm` 当 context，跑过 draft 的各层，得到 `output_hidden`。
5. 计算 `target_ids`（标签）、`eval_mask`、可选的 `aligned_target_logits`（用 target 末层 hidden 算的目标 logits，用于蒸馏）。
6. `lm_head` 出 `draft_logits`，若有 `markov_head` 则叠加偏置（`apply_block_logits`，markov_head.py:43）。
7. 若有 `confidence_head` 则预测接受率。
8. 返回 `DSparkForwardOutput`（common.py:11）。

### loss 计算 — `compute_dspark_loss`（loss.py:255）

核心在 `_collect_local_terms`（loss.py:90），三个加权项：

- **CE loss**：`draft_logits` 对 `target_ids` 的交叉熵（loss.py:112），权重 `ce_loss_alpha=0.1`
- **L1 loss**：draft 与 target 概率分布的 L1 距离（蒸馏，`_compute_local_l1_term`，loss.py:73），权重 `l1_loss_alpha=0.9`
- **Confidence loss**：接受率预测的 BCE（loss.py:157），权重 `confidence_head_alpha=1.0`
- 另带按 block 内位置的衰减权重 `loss_decay_gamma`（`_build_loss_weight_mask`，loss.py:25），以及跨进程 all-reduce 分母做全局归一（loss.py:11、`_build_loss` loss.py:227）。

## 小结（对照三个核心问题）

| 关注点 | 位置 |
|---|---|
| 调用入口 / 调度 | `train.py:31-45` → `BaseTrainer.train()` `base_trainer.py:348` |
| 模型结构 | `deepspec/modeling/dspark/qwen3/modeling.py`（`Qwen3DSparkModel` 等）+ `markov_head.py` + `common.py` |
| forward | `Qwen3DSparkModel.forward` `modeling.py:388`（经 `run_batch` `dspark_trainer.py:25` 调用） |
| loss 计算 | `deepspec/modeling/dspark/loss.py:255` `compute_dspark_loss` |

## 补充说明

训练读取的是**预先缓存好的 target hidden states**（`CacheDataset` / `CacheCollator`，batch 里的 `target_hidden_states`、`target_last_hidden_states`），并不在训练时跑 target 大模型。target 模型只在 `build_models` 里被加载一次，用于拷贝并冻结 embedding 和 `lm_head`。

> Gemma4 的训练链路与之对称：`Gemma4DSparkTrainer`（dspark_trainer.py:42）+ `deepspec/modeling/dspark/gemma4/`，loss 与 forward 复用同一套逻辑。
