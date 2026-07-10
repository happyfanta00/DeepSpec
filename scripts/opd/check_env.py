#!/usr/bin/env python3
"""DSpark-OPD 环境自检 (S0 smoke-test)。

验证 ~/.venv/dspark-opd 中「DeepSpec 依赖 + verl 0.7.0 运行时子集」均可用：
  1) 关键包版本（torch / transformers / numpy）与 GPU 数目
  2) DeepSpec 侧关键 import（草稿模型 / eval rollout 算子 / 缓存数据集）
  3) verl 0.7.0 侧关键 import（token_reward_direct advantage / 3D policy loss /
     ActorRolloutRefWorker / RewardModelWorker）——这些经核实不硬依赖 vllm/flash-attn

用法（可由用户手动执行；从任意目录皆可）：
    PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec:/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl \
        ~/.venv/dspark-opd/bin/python /home/ec2-user/efs_data/workspace/DeepSpec/scripts/opd/check_env.py

PYTHONPATH 说明：
  - DeepSpec 根目录：用于 `import deepspec.*`（第 2 步）。
  - third_party/verl：用于 `import verl.*` 与 recipe（第 3 步）。verl 已 editable 安装，
    单看 `import verl` 其实不需要它在 PYTHONPATH，但保留可确保与训练启动命令一致。
  - `verl_compat`（第 3 步的兼容 shim）无需进 PYTHONPATH：脚本会自动把自身所在的
    scripts/opd 目录加入 sys.path。

结果判读：
  - 成功：末行打印 `[S0] ENV OK`，退出码 0。
  - 失败：打印 `[S0] ENV CHECK FAILED ...` 与 traceback，退出码 1。
  用 `echo $?` 查看退出码；或直接看末行是否为 `[S0] ENV OK`。

`-h` / `--help` 打印本用法。
"""
from __future__ import annotations

import os
import sys
import traceback

# 让 `import verl_compat` 可用（与本脚本同目录），无需额外设置 PYTHONPATH。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 期望版本（见 setup_env.sh 的关键决策：numpy 走 1.26 而非 DeepSpec 的 2.4.4）
EXPECT_TORCH = "2.9.1"
EXPECT_TRANSFORMERS = "5.10.2"
EXPECT_NUMPY_MAJOR = 1  # verl 要求 numpy<2.0


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    print("\n[S0] ENV CHECK FAILED")
    sys.exit(1)


def check_versions() -> None:
    print("### 1) 版本与 GPU")
    import torch
    import transformers
    import numpy as np

    print(f"  torch        = {torch.__version__} (cuda={torch.version.cuda})")
    print(f"  transformers = {transformers.__version__}")
    print(f"  numpy        = {np.__version__}")
    ngpu = torch.cuda.device_count()
    print(f"  cuda gpus    = {ngpu}")

    if not torch.__version__.startswith(EXPECT_TORCH):
        _fail(f"torch 版本应为 {EXPECT_TORCH}.x，实际 {torch.__version__}")
    if transformers.__version__ != EXPECT_TRANSFORMERS:
        _fail(f"transformers 应为 {EXPECT_TRANSFORMERS}，实际 {transformers.__version__}")
    if int(np.__version__.split(".")[0]) != EXPECT_NUMPY_MAJOR:
        _fail(f"numpy 主版本应为 {EXPECT_NUMPY_MAJOR}.x（verl 要求 <2.0），实际 {np.__version__}")
    if ngpu < 1:
        _fail("未检测到 CUDA GPU")
    print("  [OK] 版本与 GPU 符合预期")


def check_deepspec() -> None:
    print("### 2) DeepSpec 侧 import")
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel  # noqa: F401
    from deepspec.eval.dspark.draft_ops import (  # noqa: F401
        forward_dspark_draft_block,
        build_dspark_proposal,
    )
    from deepspec.modeling.dspark.markov_head import build_markov_head  # noqa: F401
    from deepspec.data.target_cache_dataset import CacheDataset  # noqa: F401
    print("  [OK] deepspec import OK "
          "(Qwen3DSparkModel / draft_ops / markov_head / CacheDataset)")


def check_verl() -> None:
    print("### 3) verl 0.7.0 侧 import（不应触发 vllm/flash-attn 硬依赖）")
    # verl 0.7.0 × transformers 5.10.2 兼容 shim：补 AutoModelForVision2Seq 别名。
    # 必须在任何 verl import 之前应用。
    from verl_compat import apply as apply_verl_compat

    patched = apply_verl_compat()
    print(f"  verl_compat shim applied      : {patched or '(none needed)'}")

    from verl.trainer.ppo.core_algos import (
        get_adv_estimator_fn,
        compute_policy_loss_vanilla,  # noqa: F401
    )
    adv = get_adv_estimator_fn("token_reward_direct")
    print(f"  token_reward_direct advantage : {adv is not None}")
    if adv is None:
        _fail("verl 未注册 token_reward_direct advantage（应在 0.7.0 fork 中存在）")

    import verl.workers.fsdp_workers as fw
    has_actor = hasattr(fw, "ActorRolloutRefWorker")
    has_reward = hasattr(fw, "RewardModelWorker")
    print(f"  ActorRolloutRefWorker         : {has_actor}")
    print(f"  RewardModelWorker             : {has_reward}")
    if not (has_actor and has_reward):
        _fail("fsdp_workers 缺 ActorRolloutRefWorker / RewardModelWorker")

    # 确认 rollout 注册表存在（S2 将往里加 dspark 条目）
    from verl.workers.rollout.base import get_rollout_class  # noqa: F401
    print("  get_rollout_class importable  : True")
    print("  [OK] verl import OK")


def main() -> None:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        sys.exit(0)
    try:
        check_versions()
        check_deepspec()
        check_verl()
    except Exception:  # noqa: BLE001
        print("\n--- traceback ---")
        traceback.print_exc()
        print("\n[S0] ENV CHECK FAILED (import/exception)")
        sys.exit(1)

    print("\n[S0] ENV OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
