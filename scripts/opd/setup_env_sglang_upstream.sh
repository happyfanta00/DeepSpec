#!/usr/bin/env bash
# =============================================================================
# DSpark-on-SGLang 环境安装脚本 (Stage-M0，UPSTREAM sglang + verl 统一 env)
#
# 用途：把「最新 upstream sglang（HEAD）」+「verl 训练栈」装进【同一个 env】，使
#   ① Stage-M1：用其【原生 DSPARK】路径做完整-draft-分布 rejection sampling 无损验证；
#   ② Stage-M2 T3 起：verl HYBRID rollout 在【本 env 内】起 sglang engine 现场投机解码
#      —— verl 的 HYBRID replica 把 sglang 作为同 env 的 Ray actor+子进程拉起（与训练 FSDP
#      共卡），故 sglang 与 verl【必须同 env 可 import】，不能再是两个隔离 env。
#   背景 & 决策：docs/opd/dspark-on-sglang-design.md §3（环境）、§4（正确性）、§5.6（T3）。
#
# ⚠️ 本脚本【由用户手动执行】，Claude 不直接操作 python 环境。
#     建议逐 STEP 跑，观察每步输出；任一 STEP 报错就停下把 traceback 贴回。
#
# ⚠️ 关键事实（务必先读）：
#   - upstream 原生完整 DSPARK 链路（models/dspark.py + speculative/dspark_components/
#     + DSparkWorkerV2）【只在 HEAD，未进任何 release tag】。故只能用 HEAD 源码。
#     （纯 Triton 的 reject_sampling.py 从 v0.5.14 进 release，但 worker/verify 没有。）
#   - upstream HEAD 依赖很激进：torch==2.11.0 / transformers==5.12.1 / CUDA cu13 /
#     sglang-kernel==0.4.4 / flashinfer 0.6.14 / flash-attn-4 beta / rust grpc 扩展
#     / tilelang / cutlass-dsl。本机满足 cu13（toolkit 13.0，driver 13.2）。
#   - 策略（用户已定）：先试【方案2 全量官方依赖】(STEP 3B)；若卡住 → 降级【方案1 最小
#     子集】(STEP 3A + 按 import 报错逐步补)。二者都是 --no-deps editable 装 sglang 本体。
#
# 主 env ~/.venv/dspark-opd 永久不动（纯训练回退基线）；本 env = 统一 env（sglang + verl）。
# =============================================================================
set -uo pipefail   # 注意：不 set -e，便于逐步观察；关键步失败会显式提示

# ---- 路径常量 ----
SGLANG_VENV="$HOME/.venv/dspark-opd-sglang"       # 目标 env（用户已重建为全新空 env）
PY="$SGLANG_VENV/bin/python"
DEEPSPEC_DIR="/home/ec2-user/efs_data/workspace/DeepSpec"
VERL_DIR="${DEEPSPEC_DIR}/third_party/verl"
# upstream sglang vendored as a submodule (fork happyfanta00/sglang, branch
# dspark-opd-patches). Python 包在 python/ 子目录。旧的外部 clone
# /home/ec2-user/efs_data/workspace/sglang-latest 已弃用（submodule 可用后可删）。
SGLANG_UPSTREAM="${DEEPSPEC_DIR}/third_party/sglang"
SGLANG_PKG_DIR="${SGLANG_UPSTREAM}/python"

# ---- CUDA / index ----
CUINDEX=130                                        # CUDA 13.0 → torch cu130 wheel
TORCH_INDEX="https://download.pytorch.org/whl/cu${CUINDEX}"
export CUDA_HOME=/opt/pytorch/cuda                 # 本机 toolkit（deep_gemm import 需要）
# 一次性前置（幂等）：deep_gemm 找 lib64
[ -e /opt/pytorch/cuda/lib64 ] || ln -sfn /opt/pytorch/cuda/lib /opt/pytorch/cuda/lib64 2>/dev/null || true
export PATH="$CUDA_HOME/bin:$PATH"

echo "=========================================================="
echo " DSpark-on-SGLang UPSTREAM env setup (Stage-M0)"
echo "   venv     : $SGLANG_VENV"
echo "   upstream : $SGLANG_PKG_DIR  (HEAD)"
echo "   deepspec : $DEEPSPEC_DIR"
echo "   CUDA idx : cu${CUINDEX}   CUDA_HOME=$CUDA_HOME"
echo "=========================================================="
[ -x "$PY" ] || { echo "ERROR: $PY 不存在，请先 uv venv $SGLANG_VENV --python 3.11"; exit 1; }
"$PY" --version

# -----------------------------------------------------------------------------
echo; echo "### STEP 1  —  基础工具 + torch 2.11.0 (cu${CUINDEX})"
echo "    (先钉 torch，避免后续依赖从默认 PyPI 拉错 CUDA 变体)"
uv pip install --python "$PY" -U pip setuptools wheel html5lib six
uv pip install --python "$PY" --index-url "$TORCH_INDEX" \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    || { echo ">>> torch 安装失败：确认 $TORCH_INDEX 有 cu130 wheel。贴报错。"; exit 1; }
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())" \
    || { echo ">>> torch import 失败。贴报错。"; exit 1; }

# -----------------------------------------------------------------------------
echo; echo "### STEP 2  —  sgl-kernel 预编译 wheel (cu13 → pypi sglang-kernel==0.4.4)"
echo "    (Dockerfile cu13 分支用 pypi sglang-kernel==0.4.4 --no-deps)"
# [已核验] pypi 有 sglang_kernel-0.4.4-cp310-abi3-manylinux2014_x86_64.whl
#          abi3 兼容 py3.11、x86_64 匹配本机 → 可装。
uv pip install --python "$PY" --no-deps --force-reinstall "sglang-kernel==0.4.4" \
    || { echo ">>> sglang-kernel 装失败。可能 pypi 无 cu13 abi3 wheel；贴报错，考虑降级或从 whl release 找 URL。"; }

# -----------------------------------------------------------------------------
echo; echo "### STEP 3B  —  【方案2 优先】按 upstream pyproject 全量装依赖 (BUILD_TYPE=core)"
echo "    等价 Dockerfile: pip install --extra-index-url \$TORCH_INDEX '.[<extra>]'（不含 diffusion 等）"
echo "    注意：这一步会尝试装 flashinfer[cu13]/cutlass-dsl/tilelang 等；耗时长、可能编译。"
echo "    ⚠️ 我们【不装】重的可选组（diffusion/http2/tracing/test）。BUILD_TYPE 用基础依赖。"
pushd "$SGLANG_PKG_DIR" >/dev/null
# 全量核心依赖：直接装 pyproject 的 base dependencies（不带 extras）。
# 用 --extra-index-url 让 torch 系仍从 cu130，其余从 pypi。
uv pip install --python "$PY" --extra-index-url "$TORCH_INDEX" \
    --index-strategy unsafe-best-match \
    "flashinfer_python[cu13]==0.6.14" \
    || echo ">>> [方案2 信号] flashinfer[cu13] 装失败——这是最常见卡点，可先跳过 flashinfer（DSPARK verify 用 Triton，不强依赖 flashinfer softmax），走 STEP 3A 最小子集。"
popd >/dev/null

# -----------------------------------------------------------------------------
echo; echo "### STEP 3A  —  【方案1 兜底】最小子集：只装 DSPARK server 真正 import 的包"
echo "    (纯 Python + Triton 路径。DSPARK verify 的 SoftmaxTemp 有 torch/triton 回退，"
echo "     flashinfer 不可用时自动降级；见 dspark_accept.py:247-296。)"
# [已核验] DSPARK spec 链（dspark_components/reject_sampling/dflash_utils/spec_info）
#          的第三方 import 仅 torch/triton/msgspec + stdlib，不直接依赖
#          flashinfer/flash_attn/deep_gemm/cutlass/tilelang。但 `import sglang` 本体
#          会拉起整个 srt import 链——下面这批覆盖常见启动依赖；若 STEP 5 仍报某模块
#          缺失，按报错补装即可（这是预期的逐步收敛，不是失败）。
# 这批是从 dspark_verify.py / dspark_worker_v2.py / server 启动链的 import 提取的核心运行时依赖。
# 若 STEP 3B 已装全，这步多为 no-op；若 3B 卡住，这步保证最小可跑。
# [已实测通过 2026-07-15] 下面这批是实际让 `import sglang` + 原生 DSPARK 模块全部加载
#   所需的最小子集（含逐步收敛时补的 pybase64/IPython/gguf/openai-harmony 等）。
#   完整冻结见 docs/opd/pip-freeze-sglang-upstream.txt（168 包）。
uv pip install --python "$PY" \
    "transformers==5.12.1" \
    numpy \
    triton \
    msgspec \
    einops \
    "pydantic" \
    "fastapi" "uvicorn" "uvloop" "orjson" "requests" "aiohttp" \
    "pyzmq>=25.1.2" \
    "psutil" "setproctitle" "py-spy" \
    "prometheus-client>=0.20.0" \
    "packaging" "tqdm" "scipy" \
    "sentencepiece" "tiktoken" "blobfile" \
    "compressed-tensors" \
    "cuda-python>=13.0" \
    "outlines==0.1.11" "xgrammar==0.2.1" "interegular" "llguidance>=0.7.11,<0.8.0" \
    "partial_json_parser" "python-multipart" "pillow" "datasets" \
    "torchao" "torch_memory_saver>=0.0.9.post1" \
    "pybase64" "IPython" "gguf" "openai==2.6.1" "openai-harmony==0.0.4" \
    "anthropic>=0.20.0" "modelscope" "ninja" "hf_transfer" "pybind11" \
    || echo ">>> 最小子集里有包装失败，贴报错逐个处理。"

# -----------------------------------------------------------------------------
echo; echo "### STEP 3C  —  verl 运行时核心依赖（统一 env：本 env 同时跑 sglang engine + verl 训练）"
echo "    (Stage-M2 T3 起 verl HYBRID rollout 在【本 env 内】起 sglang，故 verl 必须与 sglang"
echo "     同 env 可 import。只装 verl core install_requires 里本 env 尚缺的纯依赖；不碰 torch/tf。)"
# verl core deps 已在本 env 的：datasets/dill/pyarrow/pandas/packaging/transformers（5.12.1）。
# 缺的（逐一装）：ray/tensordict/torchdata/codetiming/hydra-core(+omegaconf)/accelerate/peft/
#   pybind11/pylatexenc/wandb/tensorboard。deepspec 不需 pip 装（run.sh 用 PYTHONPATH，纯 python）。
# ⚠️ numpy（用户已定：保留 2.x）：verl 声明 numpy<2.0.0，但 sglang HEAD 需 numpy>=2（本 env 已
#   2.3.5）。下面这批 deps 均不 pin numpy<2，且 STEP 4B 装 verl 用 --no-deps → numpy 不动。若 verl
#   运行期真撞 numpy 2.x 不兼容点，再逐个 patch（verl 0.7 多数功能在 numpy 2.x 下可跑）。
# ⚠️ tensordict：verl pin <=0.10.0；但 tensordict 有按 torch 版本编译的组件，若 0.10.0 与 torch
#   2.11 ABI 不匹配 import 失败，则放宽装最新（verl 0.7 通常兼容更高 tensordict）——报错贴回。
uv pip install --python "$PY" \
    "ray[default]>=2.41.0" \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    torchdata \
    codetiming \
    "hydra-core" omegaconf \
    accelerate peft \
    pybind11 pylatexenc \
    wandb tensorboard \
    || echo ">>> verl 运行时依赖里有包装失败，贴报错逐个处理（尤其 tensordict/torch ABI）。"
# 校验：确认关键数值后端未被这批 deps 顺带改动（torch 2.11 / tf 5.12 / numpy 2.x 应原样）
"$PY" -c "import torch, transformers, numpy; print(f'  after verl-deps: torch={torch.__version__} tf={transformers.__version__} numpy={numpy.__version__}')" \
    || echo ">>> 校验失败：torch/transformers/numpy 被改动，检查上面的解析。"

# -----------------------------------------------------------------------------
echo; echo "### STEP 4  —  editable 安装 upstream sglang 本体 (--no-deps, 跳过 rust 扩展)"
echo "    (--no-deps：不让它再解析 pyproject 全量依赖，依赖已由 STEP 1-3 手工控制)"
# [已核实] pyproject 声明了一个 setuptools-rust 扩展 sglang.srt.grpc._core（gRPC server 用），
#   本机无 rust → 默认构建会报 'error: can't find Rust compiler'。
#   该扩展 DSPARK 用不到（只在 http_server.py:2540 被引用，且那里已是"未找到则降级"分支）。
#   upstream setup.py 提供官方旋钮 SGLANG_BUILD_RUST_EXTS=none → 跳过所有 rust 扩展，无需装 rust。
export SGLANG_BUILD_RUST_EXTS=none
uv pip install --python "$PY" --no-deps -e "$SGLANG_PKG_DIR" \
    || { echo ">>> sglang editable 装失败。"; \
         echo "    若仍报 rust/cargo：确认 SGLANG_BUILD_RUST_EXTS=none 已 export 到 build 子进程"; \
         echo "    （uv 默认 build isolation 会继承父环境变量；若没生效可加 --no-build-isolation-package sglang）。"; \
         echo "    其它报错贴回 Claude。"; }

# -----------------------------------------------------------------------------
echo; echo "### STEP 4B  —  editable 安装 verl 本体 (--no-deps, 统一 env)"
echo "    (--no-deps：依赖已由 STEP 3C 手工装齐，绝不让 verl 解析 install_requires 顺带改"
echo "     torch/transformers/numpy。verl 是纯 python，torch 2.11 下 editable 装无需重编。)"
uv pip install --python "$PY" --no-deps -e "$VERL_DIR" \
    || echo ">>> verl editable 装失败，贴报错。"
# deepspec 不 pip 装：run.sh 用 PYTHONPATH=$DEEPSPEC_DIR:$VERL_DIR 提供（纯 python）。

# -----------------------------------------------------------------------------
echo; echo "### STEP 5  —  冒烟：统一 env 同时可 import sglang(原生 DSPARK) + verl 训练栈 + deepspec"
echo "    (Stage-M2 T3 起 verl HYBRID rollout 在本 env 内起 sglang，故二者必须同 env 共存。)"
# deepspec 靠 PYTHONPATH，故 smoke 也带上（与 run.sh 一致）；verl recipe __init__ 触发 rollout 注册。
PYTHONPATH="$DEEPSPEC_DIR:$VERL_DIR:${PYTHONPATH:-}" "$PY" - <<'PYEOF'
import sys, traceback
# ⚠️ 顺序关键：先 import recipe.dspark_opd —— 它的 __init__ 装两个 compat shim
#   （transformers AutoModelForVision2Seq 别名 + sglang HEAD _launch_subprocesses 3-tuple 适配），
#   必须在任何 verl.workers.rollout / sglang_rollout import 之前跑（训练侧 task_runner 亦如此保证）。
ok = True
try:
    import recipe.dspark_opd  # noqa: F401  (compat shims + rollout registration)
    print(f"  [OK] import recipe.dspark_opd (shims: tf={recipe.dspark_opd._COMPAT_PATCHED} "
          f"sglang_launch={recipe.dspark_opd._SGLANG_LAUNCH_SHIMMED})")
except Exception as e:
    ok = False
    print(f"  [FAIL] import recipe.dspark_opd: {e!r}")
    traceback.print_exc(limit=3)

# (1) upstream 原生 DSPARK 链路（Stage-M0 原有断言）
sglang_mods = [
    "sglang",
    "sglang.srt.speculative.reject_sampling",
    "sglang.srt.speculative.dspark_components.dspark_verify",
    "sglang.srt.speculative.dspark_components.kernels.dspark_accept",
    "sglang.srt.models.dspark",
]
# (2) verl 训练栈 + deepspec draft modeling（统一 env 新增断言）
verl_mods = [
    "ray",
    "verl",                                  # protocol.py → ray/tensordict/torch import 链
    "verl.workers.rollout.replica",          # T3 HYBRID 用 get_rollout_replica_class
    "verl.workers.rollout.sglang_rollout.async_sglang_server",  # SGLangReplica/SGLangHttpServer（触发 _launch_subprocesses import）
    "deepspec.modeling.dspark.qwen3",        # 训练侧 draft 模型（PYTHONPATH 提供）
]
for m in sglang_mods + verl_mods:
    try:
        __import__(m)
        print(f"  [OK] import {m}")
    except Exception as e:
        ok = False
        print(f"  [FAIL] import {m}: {e!r}")
        traceback.print_exc(limit=3)
# (3) verl 能取到 sglang HYBRID replica 类（T3 前置）
try:
    from verl.workers.rollout.replica import get_rollout_replica_class
    cls = get_rollout_replica_class("sglang")
    print(f"  [OK] get_rollout_replica_class('sglang') -> {cls.__name__}")
except Exception as e:
    ok = False
    print(f"  [FAIL] get_rollout_replica_class('sglang'): {e!r}")
    traceback.print_exc(limit=3)
print("[Stage-M0] UNIFIED ENV IMPORT", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
PYEOF
RC=$?

# -----------------------------------------------------------------------------
echo; echo "### STEP 6  —  冻结依赖清单"
uv pip freeze --python "$PY" | tee "$DEEPSPEC_DIR/docs/opd/pip-freeze-sglang-upstream.txt" >/dev/null
echo "    冻结到 docs/opd/pip-freeze-sglang-upstream.txt"

echo
echo "=========================================================="
if [ "$RC" -eq 0 ]; then
  echo " Stage-M0 完成：STEP 5 打印 UNIFIED ENV IMPORT OK"
  echo "   → 本 env 同时支持 sglang 原生 DSPARK + verl 训练栈 + deepspec draft modeling。"
  echo " 下一步 Stage-M2 T3：verl HYBRID rollout 在本 env 内起 sglang（见 design.md §5.6）。"
else
  echo " Stage-M0 未完成：STEP 5 有 FAIL。把 traceback 贴回 Claude："
  echo "   - sglang_mods FAIL → 缺 STEP 3A 某包 / sgl-kernel / flashinfer 版本。"
  echo "   - verl_mods FAIL  → 缺 STEP 3C 某 verl core dep（ray/tensordict/hydra…）"
  echo "     或 tensordict 与 torch 2.11 ABI 不匹配（放宽 tensordict 上限重试）。"
fi
echo "=========================================================="
