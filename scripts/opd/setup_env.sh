#!/usr/bin/env bash
# =============================================================================
# DSpark-OPD 环境安装脚本 (S0)
#
# 用途：在干净的 uv 虚拟环境 ~/.venv/dspark-opd 中，装齐
#   「DeepSpec 运行依赖」 + 「verl 0.7.0 运行时子集」
# 供 DSpark On-Policy Distillation 训练使用（见 docs/DSpark-OPD.md 第 3 部分 S0）。
#
# ⚠️ 本脚本由用户手动执行，Claude 不直接操作 python 环境。
#     执行前请先阅读本文件顶部的「关键决策」，确认无误后再运行。
#
# 用法：
#     bash scripts/opd/setup_env.sh
#   或分步执行（推荐首次逐段运行以便观察冲突）：
#     打开本文件，按 STEP 逐条复制到终端。
#
# 前置：uv 已安装（which uv），venv 已建：uv venv ~/.venv/dspark-opd --python 3.11
# =============================================================================
set -euo pipefail

# ---- 路径常量（按你的机器已确认） ----
VENV="$HOME/.venv/dspark-opd"
PY="$VENV/bin/python"
DEEPSPEC_DIR="/home/ec2-user/efs_data/workspace/DeepSpec"
# verl 0.7.0 is VENDORED as part of this repo at third_party/verl (tracked in git).
VERL_DIR="${DEEPSPEC_DIR}/third_party/verl"

echo "=========================================================="
echo " DSpark-OPD env setup"
echo "   venv     : $VENV"
echo "   deepspec : $DEEPSPEC_DIR"
echo "   verl     : $VERL_DIR"
echo "=========================================================="

# -----------------------------------------------------------------------------
# 关键决策（安装策略说明——务必先读）
# -----------------------------------------------------------------------------
# 1) numpy 版本冲突：verl 0.7.0 要求 numpy<2.0.0，而 DeepSpec requirements.txt
#    pin 了 numpy==2.4.4。二者不可兼得。→ 本脚本以 verl 的 numpy<2.0 为准
#    （装 numpy==1.26.4）。torch 2.9 / transformers 5.10 均兼容 numpy 1.26；
#    DeepSpec 仅在数据/eval 路径用到 numpy，预期可在 1.26 下工作。
#    ✅ 该假设由 check_env.py 与后续 S1 的真实数据加载验证。
#
# 2) transformers：用 DeepSpec 要求的 5.10.2（其 Qwen3 draft modeling 直接 import
#    transformers.models.qwen3）。verl 不 pin transformers，兼容。
#
# 3) 不安装 vLLM / flash-attn 作为 rollout 引擎：
#    - 我们自研 PyTorch 块并行 rollout（§2.6.2 IP-1），不用 vLLM。
#    - verl 顶层 import 不硬依赖 vllm（get_rollout_class 用 importlib 惰性加载）。
#    - flash-attn 仅在 verl 的 remove-padding 路径惰性 import；我们将设
#      use_remove_padding=False（§2.6.4 caveat #3），故无需 flash-attn。
#    ⚠️ 若后续 import 报缺 vllm/flash_attn 符号，见文末「排障」。
#
# 4) verl 用 `pip install -e --no-deps` 安装（editable，不解析其依赖），
#    再由本脚本手工装它的运行时子集，避免 pip 把 numpy 拉回 <2 与其它约束
#    产生难解的 resolver 冲突，也避免误装 vllm。
# -----------------------------------------------------------------------------

if [ ! -x "$PY" ]; then
    echo "ERROR: $PY 不存在。请先执行： uv venv $VENV --python 3.11"
    exit 1
fi

echo
echo "### STEP 1/5  —  升级 pip 工具链"
uv pip install --python "$PY" -U pip setuptools wheel

echo
echo "### STEP 2/5  —  安装 PyTorch 2.9.1（CUDA 构建，匹配 H100 / CUDA 12.x）"
# 若默认 wheel 与本机 CUDA 不匹配，改用 --index-url 指定对应 cuXXX。
uv pip install --python "$PY" torch==2.9.1

echo
echo "### STEP 3/5  —  DeepSpec 运行依赖（注意：numpy 走 1.26，不用 DeepSpec 的 2.4.4）"
uv pip install --python "$PY" \
    "numpy==1.26.4" \
    "transformers==5.10.2" \
    "PyYAML==6.0.3" \
    "tqdm==4.67.3" \
    "tensorboard==2.20.0" \
    "matplotlib==3.10.9" \
    "triton==3.5.1" \
    "typing_extensions==4.15.0" \
    "sentencepiece==0.2.1" \
    "safetensors==0.7.0" \
    "prettytable==3.17.0" \
    "datasets==4.8.5" \
    "openai==2.6.1"

echo
echo "### STEP 4/5  —  verl 0.7.0 运行时子集（不含 vllm / flash-attn）"
# 这些是 fsdp_workers / ray_trainer / core_algos / main_ppo 顶层 import 所需。
uv pip install --python "$PY" \
    "ray[default]" \
    "hydra-core" \
    "omegaconf" \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    "codetiming" \
    "dill" \
    "pandas" \
    "pyarrow>=19.0.0" \
    "peft" \
    "accelerate" \
    "pylatexenc" \
    "latex2sympy2_extended" \
    "math_verify" \
    "torchdata" \
    "psutil" \
    "uvicorn" \
    "fastapi" \
    "packaging>=20.0" \
    "pybind11"

echo
echo "### STEP 5/5  —  editable 安装 vendored verl（--no-deps，避免依赖解析拉回 numpy<2 / vllm）"
# verl 0.7.0 已作为本仓库的一部分 vendored 在 third_party/verl（纳入 git 追踪）。
# 这里只做 editable 安装，不复制。
if [ ! -d "$VERL_DIR/verl" ]; then
    echo "ERROR: 未找到 vendored verl：$VERL_DIR"
    echo "       verl 应已作为本仓库的一部分存在于 third_party/verl。"
    exit 1
fi
uv pip install --python "$PY" --no-deps -e "$VERL_DIR"

echo
echo "=========================================================="
echo " 安装完成。冻结依赖清单："
echo "=========================================================="
uv pip freeze --python "$PY" | tee "$DEEPSPEC_DIR/docs/opd/pip-freeze.txt"

echo
echo "=========================================================="
echo " 下一步：运行环境自检"
echo "   PYTHONPATH=$DEEPSPEC_DIR:$VERL_DIR \\"
echo "     $PY $DEEPSPEC_DIR/scripts/opd/check_env.py"
echo "=========================================================="

# -----------------------------------------------------------------------------
# 排障（若 check_env.py 报 ImportError）
# -----------------------------------------------------------------------------
# A) 缺 vllm：若某处顶层 import vllm 报错，先确认是否在我们不用的 rollout 路径；
#    check_env.py 只 import core_algos / fsdp_workers / RewardModelWorker，
#    这些经核实不硬依赖 vllm。若仍报错，把具体 traceback 发给 Claude。
# B) 缺 flash_attn：应仅出现在 remove-padding 路径的惰性 import；check_env.py
#    不触发。若 import verl 顶层即报 flash_attn，同样把 traceback 发来定位。
# C) numpy 冲突/降级：若某包强制把 numpy 升到 >=2，用
#      uv pip install --python "$PY" "numpy==1.26.4"
#    重新钉住，并记录是哪个包触发的。
# -----------------------------------------------------------------------------
