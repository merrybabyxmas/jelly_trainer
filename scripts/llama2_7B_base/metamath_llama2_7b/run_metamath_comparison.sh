#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
git submodule update --init --recursive

# ============================================================
# MetaMath Instruction-Following Comparison: JELLY vs Other Methods
# ============================================================
# Dataset: MetaMath (instruction-following format)
# Methods: JELLY, LoRA, PiSSA, DoRA, BitFit
# Metric: eval_loss (lower is better)
# Output: outputs/metamath_comparison_YYYYMMDD_HHMMSS/
#         ├── results.csv
#         ├── metadata.json
#         └── logs/
# ============================================================

# GPU 설정 (병렬 실행)
GPUS="0"              # 사용할 GPU ID (예: "0,1,2,3")
PER_GPU_TASKS=1       # GPU당 동시 실행 작업 수

# 실험 설정
SEEDS="42"
# METHODS="jelly,lora,pissa,dora,bitfit"
METHODS="jelly"  # 빠른 테스트용

# Model & Dataset
MODEL="meta-llama/Llama-2-7b-hf"
DATA_PATH="fxmeng/pissa-dataset"
# MetaMath sub-task: metamath:<num_samples> or just metamath
SUB_TASK="metamath:100000"

# Training Parameters
LR=2e-4
BATCH_SIZE=4
EPOCHS=1
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
GRAD_ACCUM=1
LR_SCHEDULER="cosine"

# Data Slicing (0=no limit, >0=cap training samples for faster iteration)
# Dataset sizes: metamath=~395k samples
# 1000 samples → ~10-20 min per experiment (batch=4, grad_accum=4)
MAX_TRAIN_SAMPLES=10000

# LoRA Parameters
R=8
ALPHA=8
LORA_DROPOUT=0.0
# TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
TARGET_MODULES="q_proj,k_proj,v_proj"

# JELLY TASI Parameters
# PROBE_STEPS: number of steps for probe phase
#   - 200: ~5% of 10k-sample run (good default)
#   - 0: skip probe, start parallel (equivalent to LoRA with orthogonal init)
PROBE_STEPS=100
# PROBE_INIT_SCALE: scale for A_par after TASI (empty = auto sqrt(1/d_in))
PROBE_INIT_SCALE=""

# Test accuracy evaluation (test_acc.py)
# 학습 종료 후 test split에서 생성 기반 정확도 평가
EVAL_TEST_ACC=true
EVAL_TEST_TYPES="math,gsm8k"  # "math" 또는 "math,gsm8k" (빈 문자열=전체)
MAX_TEST_SAMPLES=0      # 0=전체 (math=5000, gsm8k=1319)
TEST_MAX_NEW_TOKENS=256

# Wandb 설정
WANDB_PROJECT="[JELLY]LLM-MetaMath"
WANDB_ENTITY="DongwooYein"

TEST_MODE=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " MetaMath Instruction-Following Comparison 실험"
echo " Model: $MODEL"
echo " Data: $DATA_PATH / $SUB_TASK"
echo " GPUs: $GPUS | Per GPU Tasks: $PER_GPU_TASKS"
echo " JELLY TASI: probe_steps=$PROBE_STEPS"
echo " Max Train Samples: $MAX_TRAIN_SAMPLES (0=all)"
echo " Eval Test Acc: $EVAL_TEST_ACC | Max Test Samples: $MAX_TEST_SAMPLES (0=all)"
echo " Epochs: $EPOCHS | LR: $LR | R: $R"
echo " 최대 동시 실행 작업 수: $(($(echo $GPUS | tr ',' '\n' | wc -l) * PER_GPU_TASKS))"
echo "============================================================"

# Build optional args
PROBE_INIT_SCALE_ARG=""
if [ -n "$PROBE_INIT_SCALE" ]; then
    PROBE_INIT_SCALE_ARG="--probe_init_scale $PROBE_INIT_SCALE"
fi

if [ "$TEST_MODE" = true ]; then
    echo "[테스트 모드]"
    python -u experiments/metamath_comparison.py \
        --gpus "$GPUS" \
        --per_gpu_tasks $PER_GPU_TASKS \
        --seeds "$SEEDS" \
        --methods "$METHODS" \
        --model "$MODEL" \
        --data_path "$DATA_PATH" \
        --sub_task "$SUB_TASK" \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --warmup_ratio $WARMUP_RATIO \
        --grad_accum $GRAD_ACCUM \
        --max_train_samples $MAX_TRAIN_SAMPLES \
        --probe_steps $PROBE_STEPS \
        $PROBE_INIT_SCALE_ARG \
        --r $R \
        --alpha $ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --target_modules "$TARGET_MODULES" \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_entity "$WANDB_ENTITY" \
        --test
else
    echo "[실험 모드]"
    EVAL_TEST_ACC_FLAG=""
    if [ "$EVAL_TEST_ACC" = true ]; then
        EVAL_TEST_ACC_FLAG="--eval_test_acc"
    fi
    python -u experiments/metamath_comparison.py \
        --gpus "$GPUS" \
        --per_gpu_tasks $PER_GPU_TASKS \
        --seeds "$SEEDS" \
        --methods "$METHODS" \
        --model "$MODEL" \
        --data_path "$DATA_PATH" \
        --sub_task "$SUB_TASK" \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --warmup_ratio $WARMUP_RATIO \
        --grad_accum $GRAD_ACCUM \
        --max_train_samples $MAX_TRAIN_SAMPLES \
        --probe_steps $PROBE_STEPS \
        $PROBE_INIT_SCALE_ARG \
        --r $R \
        --alpha $ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --target_modules "$TARGET_MODULES" \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_entity "$WANDB_ENTITY" \
        --max_test_samples $MAX_TEST_SAMPLES \
        --test_max_new_tokens $TEST_MAX_NEW_TOKENS \
        --eval_test_types "$EVAL_TEST_TYPES" \
        $EVAL_TEST_ACC_FLAG
fi

echo ""
echo "============================================================"
echo " 실험 완료!"
echo " 결과는 outputs/metamath_comparison_*/ 폴더에 저장됩니다."
echo "============================================================"
