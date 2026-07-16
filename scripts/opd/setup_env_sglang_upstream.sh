#!/usr/bin/env bash
# =============================================================================
# DSpark-on-SGLang 环境安装脚本 (Stage-M0，UPSTREAM sglang)
#
# 用途：把「最新 upstream sglang（HEAD）」装进独立 env，用其【原生 DSPARK】路径做
#   完整-draft-分布 rejection sampling 的无损验证。
#   背景 & 决策：docs/opd/dspark-on-sglang-design.md §3（环境）、§4（正确性调查）。
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
# 主 env ~/.venv/dspark-opd 永久不动（回退基线）；本 env 独立长存。
# =============================================================================
set -uo pipefail   # 注意：不 set -e，便于逐步观察；关键步失败会显式提示

# ---- 路径常量 ----
SGLANG_VENV="$HOME/.venv/dspark-opd-sglang"       # 目标 env（用户已重建为全新空 env）
PY="$SGLANG_VENV/bin/python"
DEEPSPEC_DIR="/home/ec2-user/efs_data/workspace/DeepSpec"
VERL_DIR="${DEEPSPEC_DIR}/third_party/verl"
# upstream clone（Claude 已 clone 的最新 HEAD）。Python 包在 python/ 子目录。
SGLANG_UPSTREAM="/home/ec2-user/efs_data/workspace/sglang-latest"
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
echo; echo "### STEP 5  —  冒烟：import sglang + 原生 DSPARK 模块可加载"
"$PY" - <<'PYEOF'
import sys, traceback
mods = [
    "sglang",
    "sglang.srt.speculative.reject_sampling",
    "sglang.srt.speculative.dspark_components.dspark_verify",
    "sglang.srt.speculative.dspark_components.kernels.dspark_accept",
    "sglang.srt.models.dspark",
]
ok = True
for m in mods:
    try:
        __import__(m)
        print(f"  [OK] import {m}")
    except Exception as e:
        ok = False
        print(f"  [FAIL] import {m}: {e!r}")
        traceback.print_exc(limit=2)
print("[Stage-M0] UPSTREAM DSPARK IMPORT", "OK" if ok else "FAILED")
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
  echo " Stage-M0 完成：STEP 5 打印 IMPORT OK。"
  echo " 下一步 Stage-M1：起 DSPARK server 做无损 A/B 对拍（见 findings §6）。"
else
  echo " Stage-M0 未完成：STEP 5 有 FAIL。把 traceback 贴回 Claude，"
  echo " 按缺失模块补依赖（多数是 STEP 3A 里漏的某个包，或 sgl-kernel/flashinfer 版本）。"
fi
echo "=========================================================="
