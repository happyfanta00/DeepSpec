#!/usr/bin/env bash
# DSpark-OPD recipe launcher (verl 0.7.0). Runs the fused multi-step OPD training loop
# (rollout→teacher→update in one worker RPC per step; no stage gate).
#
# Control (standard verl): epochs are primary. By default trains EPOCHS full passes over the
# dataset; total_training_steps is derived as len(train_dataloader) * EPOCHS. Set STEPS to cap at
# a fixed step count instead (overrides epochs; may stop mid-epoch).
#
# Usage (single GPU, 1 epoch = default):
#   bash recipe/dspark_opd/run.sh
# Multi-GPU, 1 epoch:
#   NGPUS=8 BATCH=64 SAVE_FREQ=100 EXP=run1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash recipe/dspark_opd/run.sh
# Fixed step count (e.g. a short smoke run):
#   NGPUS=2 BATCH=16 STEPS=3 EXP=smoke bash recipe/dspark_opd/run.sh
set -x

VENV_PY=${VENV_PY:-$HOME/.venv/dspark-opd/bin/python}
DEEPSPEC_DIR=${DEEPSPEC_DIR:-/home/ec2-user/efs_data/workspace/DeepSpec}
# Vendored verl 0.7.0 copy inside DeepSpec (installed editable). We never touch Rethink-OPD.
VERL_DIR=${VERL_DIR:-${DEEPSPEC_DIR}/third_party/verl}

NGPUS=${NGPUS:-1}
BATCH=${BATCH:-2}
# knobs: EPOCHS (primary; # full passes), STEPS (optional fixed-step cap, overrides epochs),
# checkpoint frequency, experiment name (all overridable).
EPOCHS=${EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-100}
EXP=${EXP:-dspark_opd}

# STEPS is optional: only pass trainer.total_training_steps when set (else null -> epoch-derived).
STEP_OVERRIDE=()
if [ -n "${STEPS:-}" ]; then
    STEP_OVERRIDE=(trainer.total_training_steps="$STEPS")
fi

cd "$VERL_DIR"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
PYTHONPATH="$DEEPSPEC_DIR:$VERL_DIR" \
"$VENV_PY" -m recipe.dspark_opd.main \
    --config-name dspark_trainer \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.total_epochs="$EPOCHS" \
    "${STEP_OVERRIDE[@]}" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.experiment_name="$EXP" \
    data.train_batch_size="$BATCH"
