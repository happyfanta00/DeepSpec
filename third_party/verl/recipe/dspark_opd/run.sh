#!/usr/bin/env bash
# DSpark-OPD recipe launcher (verl 0.7.0). Runs the fused multi-step OPD training loop
# (rollout→teacher→update in one worker RPC per step; no stage gate).
#
# Usage (single GPU, 1 step):
#   bash recipe/dspark_opd/run.sh
# Multi-GPU training:
#   NGPUS=8 BATCH=64 STEPS=200 SAVE_FREQ=100 EXP=run1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash recipe/dspark_opd/run.sh
set -x

VENV_PY=${VENV_PY:-$HOME/.venv/dspark-opd/bin/python}
DEEPSPEC_DIR=${DEEPSPEC_DIR:-/home/ec2-user/efs_data/workspace/DeepSpec}
# Vendored verl 0.7.0 copy inside DeepSpec (installed editable). We never touch Rethink-OPD.
VERL_DIR=${VERL_DIR:-${DEEPSPEC_DIR}/third_party/verl}

NGPUS=${NGPUS:-1}
BATCH=${BATCH:-2}
# knobs: total training steps, checkpoint frequency, experiment name (all overridable).
STEPS=${STEPS:-1}
SAVE_FREQ=${SAVE_FREQ:-1}
EXP=${EXP:-dspark_opd}

cd "$VERL_DIR"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
PYTHONPATH="$DEEPSPEC_DIR:$VERL_DIR" \
"$VENV_PY" -m recipe.dspark_opd.main \
    --config-name dspark_trainer \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.total_training_steps="$STEPS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.experiment_name="$EXP" \
    data.train_batch_size="$BATCH"
