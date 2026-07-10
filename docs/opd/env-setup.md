# DSpark-OPD 环境构建（S0 · 环境部分）

> 对应 `docs/DSpark-OPD.md` 第 3 部分 **S0**。本文件记录训练环境的安装方式与关键决策。
> **安装由用户手动执行**（Claude 不直接操作 python 环境）：产物是 `scripts/opd/setup_env.sh`，你运行它，再用 `scripts/opd/check_env.py` 自检。

## 目标环境

| 项 | 值 |
|---|---|
| venv | `~/.venv/dspark-opd`（uv 管理，Python 3.11.6） |
| DeepSpec | `/home/ec2-user/efs_data/workspace/DeepSpec` |
| verl 0.7.0（**vendored**） | `DeepSpec/third_party/verl`（`version=0.7.0.dev`，editable 安装） |
| verl 参考源（只读，不修改） | `/home/ec2-user/efs_data/workspace/Rethink-OPD/verl` |
| 机器 | 8 × H100 80GB |

> **代码位置约定**：所有 OPD 开发都在 **DeepSpec 仓库内**完成，**不修改 Rethink-OPD**。verl 0.7.0 已 **vendored 为本仓库的一部分**于 `DeepSpec/third_party/verl`（纳入 git 追踪）并 editable 安装；我们的 recipe 位于 `DeepSpec/third_party/verl/recipe/dspark_opd/`（沿用 verl 的 recipe 机制）。详见文末「vendoring 与 git」。

## 安装步骤（你执行）

```bash
# 0) 若尚未建 venv（当前已建，可跳过）
uv venv ~/.venv/dspark-opd --python 3.11

# 1) 运行安装脚本（editable 安装 vendored verl；verl 已是本 repo 的一部分，不复制）
bash /home/ec2-user/efs_data/workspace/DeepSpec/scripts/opd/setup_env.sh

# 2) 环境自检（S0 smoke-test / E2E 前置）
PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec:/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl \
    ~/.venv/dspark-opd/bin/python /home/ec2-user/efs_data/workspace/DeepSpec/scripts/opd/check_env.py
```

## S0 手动 smoke-test：`check_env.py`（你可自行执行）

这是 S0 的 smoke-test，**你也可以手动跑**来独立验证环境。命令从任意目录皆可执行：

```bash
PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec:/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl \
    ~/.venv/dspark-opd/bin/python /home/ec2-user/efs_data/workspace/DeepSpec/scripts/opd/check_env.py
echo "exit code: $?"      # 关注退出码
```

看用法：`~/.venv/dspark-opd/bin/python scripts/opd/check_env.py --help`。

**通过的判据**（两者其一即可确认）：
- 末行打印 **`[S0] ENV OK`**；
- 退出码 **`0`**（`echo $?`）。

**成功时的完整预期输出**（逐行核对；实测已确认）：

```
### 1) 版本与 GPU
  torch        = 2.9.1+cu128 (cuda=12.8)
  transformers = 5.10.2
  numpy        = 1.26.4
  cuda gpus    = 8
  [OK] 版本与 GPU 符合预期
### 2) DeepSpec 侧 import
  [OK] deepspec import OK (Qwen3DSparkModel / draft_ops / markov_head / CacheDataset)
### 3) verl 0.7.0 侧 import（不应触发 vllm/flash-attn 硬依赖）
  verl_compat shim applied      : ['AutoModelForVision2Seq']
  token_reward_direct advantage : True
  ActorRolloutRefWorker         : True
  RewardModelWorker             : True
  get_rollout_class importable  : True
  [OK] verl import OK

[S0] ENV OK
```

**它断言了什么**（失败会 `exit(1)` 并打印 traceback）：
- torch==2.9.1、transformers==5.10.2、numpy 主版本==1、CUDA GPU 数≥1；
- DeepSpec 侧 `Qwen3DSparkModel` / `draft_ops` / `markov_head` / `CacheDataset` 可 import；
- verl 侧 `token_reward_direct` advantage 已注册、`ActorRolloutRefWorker` / `RewardModelWorker` / `get_rollout_class` 存在（import 前自动应用 `verl_compat` shim）。

**失败时**：末行为 `[S0] ENV CHECK FAILED ...`、退出码 `1`，并打印 traceback；对照文末「排障」或把 traceback 发来定位。

## 关键决策（安装策略）

已核实 verl 0.7.0 源码后确定：

1. **numpy 版本冲突裁决 → 用 numpy 1.26**
   verl 要 `numpy<2.0.0`，DeepSpec `requirements.txt` pin `numpy==2.4.4`，二者互斥。
   以 verl 的 `<2.0` 为准（装 `numpy==1.26.4`）：torch 2.9 / transformers 5.10 均兼容 numpy 1.26，DeepSpec 仅在数据/eval 路径用 numpy，预期可用。**由 check_env.py + S1 真实数据加载验证此假设。**

2. **transformers 用 5.10.2**（DeepSpec 的 Qwen3 draft modeling 直接 `import transformers.models.qwen3`）；verl 不 pin transformers，兼容。

3. **不装 vLLM / flash-attn**（作为 rollout 引擎）：
   - 自研 PyTorch 块并行 rollout（§2.6.2 IP-1），不用 vLLM；
   - 已核实 verl 顶层 import **不硬依赖 vllm**（`get_rollout_class` 用 `importlib` 惰性加载；`verl/__init__`、`fsdp_workers.py`、`core_algos.py`、`main_ppo.py` 顶层均无 vllm/flash import）；
   - flash-attn 仅在 remove-padding 路径**惰性 import**，我们将设 `use_remove_padding=False`（§2.6.4 caveat #3）。

4. **verl 用 `-e --no-deps` 装**，运行时依赖由脚本手工列装——避免 pip 依赖解析把 numpy 拉回 `<2` 冲突或误装 vllm。

5. **logger 用 console/tensorboard**（verl `Tracking` 支持），无需 swanlab/wandb 登录。

## 已核实的 verl import 事实（支撑上述决策）

- `verl/workers/rollout/base.py:81-101`：`_ROLLOUT_REGISTRY` 用 `importlib.import_module` 惰性加载 vllm rollout，未选 vllm 时不 import。
- `verl/workers/fsdp_workers.py` 顶层 import（L18-93）无 vllm/flash-attn。
- `verl/utils/torch_functional.py`、`verl/utils/attention_utils.py`：flash-attn 均在函数体内 `try` 块惰性 import。
- `verl/trainer/ppo/core_algos.py`：`get_adv_estimator_fn`（L134）、`compute_policy_loss_vanilla`（L1058）存在。
- `verl/utils/tracking.py`：支持 `console/tensorboard/wandb/swanlab/mlflow`。

## 已知兼容性问题与修复

### transformers 5.10.2 移除了 `AutoModelForVision2Seq`（已修复）

**现象**：`check_env.py` 第 3 步 import verl 时报
```
ImportError: cannot import name 'AutoModelForVision2Seq' from 'transformers'
  （verl/utils/model.py:28 → from transformers import (..., AutoModelForVision2Seq, ...)）
```

**根因**：verl 0.7.0 基于较老 transformers 编写，用到 `AutoModelForVision2Seq`；
transformers 5.10.2 已将其更名/合并为 `AutoModelForImageTextToText`。该符号仅用于
**视觉多模态**路径（`verl/utils/model.py`、`verl/model_merger/base_model_merger.py`，
共 5 处），DeepSpec 纯文本 OPD **完全用不到**。

**排查结论**：扫描 verl 全库所有 `from transformers import` 符号，在 5.10.2 下**仅此 1 个缺失**
——API 漂移面极小。打别名后实测 verl 四条核心 import 链
（`core_algos` / `fsdp_workers` / `rollout.base` / `trainer.main_ppo`）全部通过、无后续错误。

**修复（最小侵入）**：新增运行时兼容 shim `scripts/opd/verl_compat.py`，在**任何 verl import 之前**
调用 `apply()`，给 transformers 补别名 `AutoModelForVision2Seq = AutoModelForImageTextToText`。
- ✅ 不改 verl 源码（fork 保持干净）
- ✅ 不降 transformers（DeepSpec 的 Qwen3 draft modeling 依赖 5.10.2）
- `check_env.py` 已在 import verl 前调用它；后续 recipe 入口 `main.py` 顶部同样先调用。

若将来 verl 或 transformers 升级又出现新的缺失符号，往 `verl_compat._ALIASES` 增补 `(旧名, 新名)` 即可。

### recipe 侧 shim 必须在 Ray worker 中生效（已修复）

`scripts/opd/verl_compat.py` 只在**驱动进程**生效。verl 用 Ray，actor 会在**独立 worker 进程**里被重建，重建时 Ray 直接 import actor 的定义模块——若该模块是 `__main__`（`python -m recipe.dspark_opd.main` 时 `DSparkTaskRunner.__module__ == "__main__"`），worker 无法干净重 import，shim 不会在 verl import transformers 之前运行 → worker 侧仍报 `AutoModelForVision2Seq`。

**修复**：
1. shim 内联进 recipe 包 `recipe/dspark_opd/__init__.py`（import 包即应用；worker 重建 actor 时必然先跑包 `__init__`）。
2. `DSparkTaskRunner` 定义在真实子模块 `recipe/dspark_opd/task_runner.py`（不是 `main.py`/`__main__`），使 Ray 在 worker 中通过 `import recipe.dspark_opd.task_runner` 重建它，从而**保证先执行包 `__init__` 的 shim + rollout 注册**。`main.py` 仅作瘦入口。

### 阶段-gate 的清退不能用 `sys.exit()`（已修复）

在 Ray actor 里 `sys.exit(0)` 会杀死 worker，驱动端报 `ActorDiedError`（rc=1），即便 worker 实际以 0 退出。**修复**：gate 改为 `raise GateStop(phase)`，由 `TaskRunner.run()` 捕获后正常 `return`，actor 任务成功完成，驱动端得 rc=0。

## vendoring 与 git（verl 是本仓库的一部分）

**决策**：verl 0.7.0 已 **vendored 为 DeepSpec 仓库的一部分**，位于 `DeepSpec/third_party/verl`，**纳入 git 追踪**。所有开发在此进行，Rethink-OPD 保持只读、不修改。理由：既保留 verl 的 recipe 机制（recipe 需以 `recipe.dspark_opd.*` 被 import，故须在 verl 树内），又让全部代码（含 verl 与我们的 recipe）活在 DeepSpec 仓库、可复现、git 归属清晰。

- **复制是一次性历史操作**（已完成，`rsync` 自 Rethink-OPD/verl，14M，无独立 `.git`，不引入嵌套 git）。此后 verl 就是本 repo 的普通目录；**`setup_env.sh` 不再复制**，只做 editable 安装。
- **editable 安装**：`uv pip install --no-deps -e DeepSpec/third_party/verl` → `import verl` 解析到 `DeepSpec/third_party/verl/verl/`。
- **git 追踪**：`third_party/verl` **整体纳入追踪**（1132 文件），仅靠全局规则把 `__pycache__` 与 `*.egg-info`（含 editable 生成的 `verl.egg-info`）排除，不额外 ignore verl。
- **启动路径**：`PYTHONPATH=DeepSpec:DeepSpec/third_party/verl`，`python -m recipe.dspark_opd.main`。已实测 S0 `data` gate rc=0、日志零 Rethink-OPD 引用。

## 产物

- `scripts/opd/setup_env.sh` — 安装脚本（你执行；含 vendoring 复制 + editable 安装）
- `scripts/opd/check_env.py` — 环境自检（S0 smoke-test，import verl 前先应用 shim）
- `scripts/opd/verl_compat.py` — verl 0.7.0 × transformers 5.10.2 兼容 shim（recipe 内已内联同款）
- `DeepSpec/third_party/verl/` — vendored verl（纳入 git 追踪；`__pycache__`/`*.egg-info` 由全局规则排除）
- `docs/opd/pip-freeze.txt` — 安装后由脚本自动写出的依赖冻结清单

## 排障

- **import 报缺 vllm**：check_env.py 只 import 经核实无 vllm 依赖的模块；若仍报错，把 traceback 发来定位（可能需在 recipe 入口惰性绕过）。
- **import 报缺 flash_attn**：应仅出现在 remove-padding 路径；若顶层即报，把 traceback 发来。
- **numpy 被某包升到 >=2**：`uv pip install --python ~/.venv/dspark-opd/bin/python "numpy==1.26.4"` 重新钉住，并记录触发包。
