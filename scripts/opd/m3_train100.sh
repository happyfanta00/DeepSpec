#!/usr/bin/env bash
# Stage-M3 — DSpark-OPD 正式训练（sglang DSPARK 现场 rollout + 融合 train_step）100 步。
#
# 复用 Stage-M2 §5.10 集成验证通过的配置：
#   - DSPARK_SGLANG_ROLLOUT=1     upstream DSPARK sglang tp=8 现场投机解码 rollout（做法X 纯 prompt 源）
#   - DSPARK_SGLANG_KV_OFFLOAD=1  错峰：rollout 用大 KV 池、训练前 release(kv) 还显存（weights 常驻）
#   - DSPARK_SGLANG_MEM_FRACTION=0.55   sglang 池占比（rollout 态 sglang≈48G + 训练模型共存 ≤80G）
#   - dspark_max_anchors_cap=64（config 内）  封顶 [B,A,blk,V] 最坏张量，防长 response OOM
#
# ⚠️ expandable_segments 与 KV-offload 的 TorchMemorySaver 互斥，勿设 PYTORCH_CUDA_ALLOC_CONF。
# ⚠️ 每个 ckpt ~40G。SAVE_FREQ=25 => 4 个中间 + 1 个最终 ≈ 5×40G=200G（/efs 有 ~936G 空间，安全）。
#
# 用法（默认 8 卡 100 步）：
#   bash scripts/opd/m3_train100.sh
# 覆盖示例：
#   STEPS=200 SAVE_FREQ=50 EXP=m3_run200 BATCH=8 bash scripts/opd/m3_train100.sh
#   DSPARK_VIZ=1 bash scripts/opd/m3_train100.sh          # 额外打印 rollout/block/teacher-draft 中间量
set -euo pipefail

DEEPSPEC_DIR=${DEEPSPEC_DIR:-/home/ec2-user/efs_data/workspace/DeepSpec}
VERL_DIR=${VERL_DIR:-${DEEPSPEC_DIR}/third_party/verl}
VENV_PY=${VENV_PY:-$HOME/.venv/dspark-opd-sglang/bin/python}

# ---- 训练规模（可用环境变量覆盖）----
NGPUS=${NGPUS:-8}
BATCH=${BATCH:-8}                 # 必须 % NGPUS == 0（dp=NGPUS，每 rank 整数个 prompt）
STEPS=${STEPS:-100}               # 固定步数（覆盖 epoch 派生）
SAVE_FREQ=${SAVE_FREQ:-25}        # 每 N 步存一次（~40G/个；最终步总会存）
EXP=${EXP:-m3_run100}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ---- sglang rollout 错峰（Stage-M2 验证过的组合）----
export DSPARK_SGLANG_ROLLOUT=${DSPARK_SGLANG_ROLLOUT:-1}
export DSPARK_SGLANG_KV_OFFLOAD=${DSPARK_SGLANG_KV_OFFLOAD:-1}
export DSPARK_SGLANG_MEM_FRACTION=${DSPARK_SGLANG_MEM_FRACTION:-0.55}

# ---- 日志落盘（带时间戳；用外部 EPOCH 环境秒数以避免脚本内取时间）----
LOG_DIR=${LOG_DIR:-${DEEPSPEC_DIR}/logs/opd}
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${EXP}.log"

echo "=================================================================="
echo "[m3_train100] Stage-M3 正式训练"
echo "  NGPUS=$NGPUS BATCH=$BATCH STEPS=$STEPS SAVE_FREQ=$SAVE_FREQ EXP=$EXP"
echo "  KV_OFFLOAD=$DSPARK_SGLANG_KV_OFFLOAD MEM_FRACTION=$DSPARK_SGLANG_MEM_FRACTION"
echo "  ckpt -> ${VERL_DIR}/checkpoints/dspark_opd/${EXP}/global_step_*"
echo "  log  -> ${LOG_FILE}"
echo "=================================================================="

STEP_OVERRIDE=()
if [ -n "${STEPS:-}" ]; then
    STEP_OVERRIDE=(trainer.total_training_steps="$STEPS")
fi

cd "$VERL_DIR"
set -x
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHONPATH="$DEEPSPEC_DIR:$VERL_DIR" \
"$VENV_PY" -m recipe.dspark_opd.main \
    --config-name dspark_trainer \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.total_epochs=1 \
    "${STEP_OVERRIDE[@]}" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.experiment_name="$EXP" \
    trainer.project_name=dspark_opd \
    data.train_batch_size="$BATCH" \
    2>&1 | tee "$LOG_FILE"
