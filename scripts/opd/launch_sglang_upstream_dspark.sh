#!/usr/bin/env bash
# =============================================================================
# 启动 UPSTREAM sglang 的原生 DSPARK server（Stage-M1）。由用户手动执行（长时进程）。
#
# 用途：起 upstream 原生 DSPARK 投机解码 server，用其【完整-draft-分布 rejection
#   sampling】（reject_sampling.py: min(1,p/q)+残差）产出 response，供 Stage-M1 与本
#   repo base_evaluator 的 rejection sampling 做 A/B 无损对拍。
#   背景：docs/opd/dspark-on-sglang-design.md §5（Stage-M1）、§5.5（起 server 关键事实）。
#
# 要点：
#   - 用 UPSTREAM env（HEAD，torch2.11/cu130）。
#   - DSPARK gamma 由 ckpt config 自动读（block_size=7）→ num_draft_tokens=8，无需手动传。
#   - 默认 flashinfer attention backend（target + draft）；triton 在 spec-decode 长序列
#     GEMM 会 CUBLAS_STATUS_EXECUTION_FAILED 崩，见 design.md §5.5。
#
# 用法：
#   bash scripts/opd/launch_sglang_upstream_dspark.sh dspark      # DSPARK 投机解码
#   bash scripts/opd/launch_sglang_upstream_dspark.sh baseline    # 纯 target（对拍基准）
#
# 环境变量（覆盖默认）：
#   MODEL   target（默认 Qwen/Qwen3-4B）
#   DRAFT   DSpark draft ckpt（默认 = 用户提供的 block7 markov ckpt）
#   PORT    端口（默认 30000）
#   GPU     CUDA_VISIBLE_DEVICES（默认 0）
#   MEM     mem-fraction-static（默认 0.85）
#   ATTN    attention backend（默认 triton）
#   PY      python（默认 upstream env）
# =============================================================================
set -euo pipefail

MODE="${1:-dspark}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT:-/mnt/scratch/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest}"
PORT="${PORT:-30000}"
GPU="${GPU:-0}"
MEM="${MEM:-0.8}"
# attention backend：默认 flashinfer（0.6.14 已装、最成熟；triton 在 spec-decode 长序列
# 路径会触发 CUBLAS_STATUS_EXECUTION_FAILED 崩溃，见 findings §Stage-M1）。
# 需要纯 eager 定位问题时：ATTN=triton NO_CUDA_GRAPH=1。
ATTN="${ATTN:-flashinfer}"
NO_CUDA_GRAPH="${NO_CUDA_GRAPH:-}"   # 置 1 则加 --disable-cuda-graph（关 target+draft 所有 graph）
PY="${PY:-$HOME/.venv/dspark-opd-sglang/bin/python}"

# ---- 自包含 CUDA env（deep_gemm / sgl_kernel import 需要 toolkit）----
export CUDA_HOME="${CUDA_HOME:-/opt/pytorch/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
# rust ext 已在装包时跳过；运行期无关。

[ -e "$CUDA_HOME/lib64" ] || echo "[warn] $CUDA_HOME/lib64 缺失，建议：ln -sfn $CUDA_HOME/lib $CUDA_HOME/lib64"

echo "[launch-upstream] mode=$MODE model=$MODEL draft=$DRAFT port=$PORT gpu=$GPU attn=$ATTN"
echo "[launch-upstream] python=$PY"
echo "[launch-upstream] nvcc=$(command -v nvcc || echo MISSING)"

COMMON=(
  --model-path "$MODEL"
  --port "$PORT"
  --mem-fraction-static "$MEM"
  --attention-backend "$ATTN"
  --trust-remote-code
)
# 纯 eager 定位开关：--disable-cuda-graph 关掉 target+draft 所有 cuda graph
# （DSPARK draft worker 只认 disable_cuda_graph 总开关，见 dspark_worker_v2.py:297）。
[ -n "$NO_CUDA_GRAPH" ] && COMMON+=( --disable-cuda-graph )

case "$MODE" in
  baseline)
    # 纯 target serving，作为无损对拍的"同引擎纯 target"基准（findings §3.3 方法论）。
    exec "$PY" -m sglang.launch_server "${COMMON[@]}"
    ;;
  dspark)
    # 原生 DSPARK：gamma 自动从 draft ckpt config 读（block_size=7 → num_draft_tokens=8）。
    exec "$PY" -m sglang.launch_server "${COMMON[@]}" \
        --speculative-algorithm DSPARK \
        --speculative-draft-model-path "$DRAFT" \
        --speculative-draft-attention-backend "$ATTN"
    ;;
  *)
    echo "未知 mode: $MODE（应为 dspark / baseline）" >&2
    exit 2
    ;;
esac
