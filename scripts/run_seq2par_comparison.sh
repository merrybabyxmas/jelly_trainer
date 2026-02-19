#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
export CUDA_HOME="/home/dongwoo43/miniconda3/envs/jelly"

PYTHON="/home/dongwoo43/miniconda3/envs/jelly/bin/python"

# ============================================================
# dynamic vs seq2par vs parallel 비교 실험
# 3 experiments:
#   1) jelly_mode=dynamic (auto-switch via Gradient Coherence)
#   2) jelly_mode=seq2par, switch_epoch=0.1
#   3) jelly_mode=parallel
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

git submodule update --init --recursive

# 공통 설정
GPU="0"
SEED=42
TASK="arc_challenge"
MODEL="meta-llama/Llama-2-7b-hf"
LR=3e-4
BATCH_SIZE=4
EPOCHS=5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
GRAD_ACCUM=1
R=8
ALPHA=8
LORA_DROPOUT=0.1
TARGET_MODULES="q_proj,k_proj,v_proj"
WANDB_PROJECT="[JELLY]seq2par-comparison"
WANDB_ENTITY="DongwooYein"

COMMON_ARGS="--task $TASK --adapter jelly --model $MODEL --seed $SEED \
    --learning_rate $LR --batch $BATCH_SIZE --epochs $EPOCHS \
    --weight_decay $WEIGHT_DECAY --warmup_ratio $WARMUP_RATIO \
    --grad_accum $GRAD_ACCUM --r $R --alpha $ALPHA \
    --lora_dropout $LORA_DROPOUT --target_modules $TARGET_MODULES \
    --wandb_project $WANDB_PROJECT --wandb_entity $WANDB_ENTITY"

echo "============================================================"
echo " dynamic vs seq2par vs parallel 비교 실험 시작"
echo " Task: $TASK | Model: $MODEL | GPU: $GPU"
echo "============================================================"

# Experiment 1: dynamic (auto-switch via Gradient Coherence)
echo ""
echo ">>> [1/3] dynamic (auto-switch via Gradient Coherence)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u train_CS.py $COMMON_ARGS \
    --jelly_mode dynamic

# Experiment 2: seq2par, switch_epoch=0.1
echo ""
echo ">>> [2/3] seq2par, switch_epoch=0.1"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u train_CS.py $COMMON_ARGS \
    --jelly_mode seq2par --switch_epoch 0.1

# Experiment 3: parallel
echo ""
echo ">>> [3/3] parallel"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u train_CS.py $COMMON_ARGS \
    --jelly_mode parallel --switch_epoch 0

echo ""
echo "============================================================"
echo " 모든 실험 완료!"
echo " wandb project: $WANDB_PROJECT"
echo "============================================================"
