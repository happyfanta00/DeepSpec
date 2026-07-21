#!/usr/bin/env bash
# Stage-M3 DSpark-OPD 正式训练 — 全部参数 hard-code，不读任何环境变量。
set -x

# 干掉可能残留的旧 ray 集群（否则会连到旧的、只有 1 GPU 的集群 -> "Total available GPUs 1.0"）
/home/ec2-user/.venv/dspark-opd-sglang/bin/ray stop --force || true
sleep 3

cd /home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH=/home/ec2-user/efs_data/workspace/DeepSpec:/home/ec2-user/efs_data/workspace/DeepSpec/third_party/verl \
DSPARK_SGLANG_ROLLOUT=1 \
DSPARK_SGLANG_KV_OFFLOAD=1 \
DSPARK_SGLANG_MEM_FRACTION=0.65 \
PYTHONUNBUFFERED=1 \
/home/ec2-user/.venv/dspark-opd-sglang/bin/python -m recipe.dspark_opd.main \
    --config-name dspark_trainer \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2000 \
    trainer.save_freq=1000 \
    trainer.project_name=dspark_opd \
    trainer.experiment_name=baseline-2k-steps-run2 \
    data.train_batch_size=64 \
    2>&1 | tee /home/ec2-user/efs_data/workspace/DeepSpec/logs/opd/baseline-run2.log
