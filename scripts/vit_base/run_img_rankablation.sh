#!/bin/bash
# ============================================================
# Image Classification Rank Ablation Experiment
# ============================================================
# ViT-B/16에서 다양한 rank에 대한 lora, lava, lava_fullweight 비교
#
# 출력 구조:
#     experiments/outputs/img_rankablation_YYYYMMDD_HHMMSS/
#         ├── results.csv
#         ├── metadata.json
#         └── logs/
# ============================================================

# GPU 설정 (병렬 실행)
GPUS="0"
PER_GPU_TASKS=3

# 실험 설정
SEEDS="42"
# TASKS="dtd,eurosat,gtsrb,resisc45,sun397,svhn"
TASKS="eurosat,gtsrb,resisc45,sun397,svhn"
METHODS="lora,lava,lava_fullweight"
RANKS="4,8,12,16"

# Training Parameters
LR=1e-4
BATCH_SIZE=32
EPOCHS=15
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1

# LAVA Lambda Config
LAMBDA_VIB=0.0
LAMBDA_LATENT_STAB=0.0

# Data Ratio (1-100, percentage of training data to use)
TRAIN_DATA_RATIO=100

# Wandb 설정
WANDB_PROJECT="IMG-RankAblation"

TEST_MODE=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " Image Rank Ablation Experiment"
echo "============================================================"
echo " GPUs: $GPUS (동시 작업 수: $PER_GPU_TASKS)"
echo " Methods: $METHODS"
echo " Ranks: $RANKS"
echo " Tasks: $TASKS"
echo " Seeds: $SEEDS"
echo " Epochs: $EPOCHS"
echo " Train Data Ratio: $TRAIN_DATA_RATIO%"
echo "============================================================"

if [ "$TEST_MODE" = true ]; then
    echo "[테스트 모드]"
    python -u experiments/img_rankablation.py \
        --gpus "$GPUS" \
        --per_gpu_tasks $PER_GPU_TASKS \
        --seeds "$SEEDS" \
        --tasks "$TASKS" \
        --methods "$METHODS" \
        --ranks "$RANKS" \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --warmup_ratio $WARMUP_RATIO \
        --lambda_vib $LAMBDA_VIB \
        --lambda_latent_stab $LAMBDA_LATENT_STAB \
        --train_data_ratio $TRAIN_DATA_RATIO \
        --wandb_project "$WANDB_PROJECT" \
        --test
else
    echo "[실험 모드]"
    python -u experiments/img_rankablation.py \
        --gpus "$GPUS" \
        --per_gpu_tasks $PER_GPU_TASKS \
        --seeds "$SEEDS" \
        --tasks "$TASKS" \
        --methods "$METHODS" \
        --ranks "$RANKS" \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --warmup_ratio $WARMUP_RATIO \
        --lambda_vib $LAMBDA_VIB \
        --lambda_latent_stab $LAMBDA_LATENT_STAB \
        --train_data_ratio $TRAIN_DATA_RATIO \
        --wandb_project "$WANDB_PROJECT"
fi
