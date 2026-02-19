#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
git submodule update --init --recursive

# ============================================================
# Image Classification Comparison: JELLY vs Other Methods (병렬 GPU 실행)
# ============================================================
# Datasets:qnli  ,qqp  ,mnl
# Methods: BitFit, LoRA, AdaLoRA, DoRA, PiSSA, JELLY
# Output: outputs/img_comparison_YYYYMMDD_HHMMSS/
#         ├── results.csv
#         ├── metadata.json
#         └── logs/
# ============================================================

# GPU 설정 (병렬 실행)

GPUS="0"           # 사용할 GPU ID (예: "0,1,2,3")
PER_GPU_TASKS=2     # GPU당 동시 실행 작업 수

# 실험 설정
# SEEDS="16,33,57,67,91"
SEEDS="16"

# TASKS="cola,sst2,mrpc,stsb,qqp,mnli,qnli,rte"
TASKS="mrpc"

# METHODS="jelly,lora,pissa,bitfit"
METHODS="pissa,lora,bitfit"


# Training Parameters
LR=1e-4
BATCH_SIZE=32
EPOCHS=15
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1

# LoRA Parameters
R=8
ALPHA=8
LORA_DROPOUT=0.1
# TARGET_MODULES="query_proj,key_proj,value_proj,dense"
TARGET_MODULES="query_proj,key_proj,value_proj"


# JELLY Mode Options
# - "parallel": Start with Parallel mode (same as LoRA)
# - "sequential": Use Sequential mode throughout
# - "seq2par": Start Sequential -> Switch to Parallel (at switch_epoch)
# - "dynamic": Start Sequential -> Auto-switch via Gradient Coherence
JELLY_MODE="dynamic"

# Data Ratio (1-100, percentage of training data to use)
TRAIN_DATA_RATIO=100

# Data Slicing (0=no limit, >0=cap training samples for faster iteration)
MAX_TRAIN_SAMPLES=0

# Wandb 설정
WANDB_PROJECT="[JELLY]GLUE-comparison-fixed"

TEST_MODE=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " NLU Comparison 실험"
echo " GPUs: $GPUS | Per GPU Tasks: $PER_GPU_TASKS"
echo " JELLY Mode: $JELLY_MODE"
echo " 최대 동시 실행 작업 수: $(($(echo $GPUS | tr ',' '\n' | wc -l) * PER_GPU_TASKS))"
echo "============================================================"

if [ "$TEST_MODE" = true ]; then
    echo "[테스트 모드]"
    python -u experiments/glue_comparison.py \
        --gpus "$GPUS" \
        --per_gpu_tasks $PER_GPU_TASKS \
        --seeds "$SEEDS" \
        --tasks "$TASKS" \
        --methods "$METHODS" \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --warmup_ratio $WARMUP_RATIO \
        --jelly_mode "$JELLY_MODE" \
        --r $R \
        --alpha $ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --target_modules "$TARGET_MODULES" \
        --train_data_ratio $TRAIN_DATA_RATIO \
        --max_train_samples $MAX_TRAIN_SAMPLES \
        --wandb_project "$WANDB_PROJECT" \
        --test
else
    echo "[실험 모드]"
    python -u experiments/glue_comparison.py \
        --gpus "$GPUS" \
        --per_gpu_tasks $PER_GPU_TASKS \
        --seeds "$SEEDS" \
        --tasks "$TASKS" \
        --methods "$METHODS" \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --warmup_ratio $WARMUP_RATIO \
        --r $R \
        --jelly_mode "$JELLY_MODE" \
        --alpha $ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --target_modules "$TARGET_MODULES" \
        --train_data_ratio $TRAIN_DATA_RATIO \
        --max_train_samples $MAX_TRAIN_SAMPLES \
        --wandb_project "$WANDB_PROJECT"
fi

echo "결과는 outputs/ 폴더에 저장됩니다."
