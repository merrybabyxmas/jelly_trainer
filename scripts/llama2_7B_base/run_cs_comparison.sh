#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
git submodule update --init --recursive

# ============================================================
# Commonsense Reasoning Comparison: JELLY vs Other Methods (병렬 GPU 실행)
# ============================================================
# Datasets: PIQA, SIQA, ARC-Easy, ARC-Challenge, OpenBookQA, HellaSwag, WinoGrande
# Methods: BitFit, LoRA, AdaLoRA, DoRA, PiSSA, JELLY
# Output: outputs/commonsense_comparison_YYYYMMDD_HHMMSS/
#         ├── results.csv
#         ├── metadata.json
#         └── logs/
# ============================================================

# GPU 설정 (병렬 실행)
GPUS="0"        # 사용할 GPU ID
PER_GPU_TASKS=2       # GPU당 동시 실행 작업 수

# 실험 설정
SEEDS="42"
TASKS="piqa,siqa,arc_easy,arc_challenge,openbookqa,hellaswag,winogrande"
# TASKS="arc_challenge"  # 작은 데이터셋만
METHODS="jelly,lora,dora,pissa,bitfit"
#METHODS="dora,pissa"  # 테스트용

# Model
MODEL="meta-llama/Llama-2-7b-hf"

# Training Parameters
LR=3e-4
BATCH_SIZE=4
EPOCHS=5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
GRAD_ACCUM=1

# LoRA Parameters
R=8
ALPHA=8
LORA_DROPOUT=0.1
TARGET_MODULES="q_proj,k_proj,v_proj"

# JELLY Mode Options
# - "parallel": Start with Parallel mode (same as LoRA)
# - "sequential": Use Sequential mode throughout
# - "seq2par": Start Sequential -> Switch to Parallel (at switch_epoch)
JELLY_MODE="seq2par"

# Switch Epoch 실험 목록 (소수점 지원)
SWITCH_EPOCHS="1 0.5"

# Wandb 설정
WANDB_PROJECT="[JELLY]Llama2-dongwoo"
WANDB_ENTITY="DongwooYein"

TEST_MODE=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " Commonsense Reasoning Comparison 실험"
echo " Model: $MODEL"
echo " GPUs: $GPUS | Per GPU Tasks: $PER_GPU_TASKS"
echo " JELLY Mode: $JELLY_MODE"
echo " Switch Epochs: $SWITCH_EPOCHS"
echo " Epochs: $EPOCHS"
echo " 최대 동시 실행 작업 수: $(($(echo $GPUS | tr ',' '\n' | wc -l) * PER_GPU_TASKS))"
echo "============================================================"

for SWITCH_EPOCH in $SWITCH_EPOCHS; do
    echo ""
    echo "============================================================"
    echo " Switch Epoch: $SWITCH_EPOCH / Epochs: $EPOCHS"
    echo "============================================================"

    if [ "$TEST_MODE" = true ]; then
        echo "[테스트 모드]"
        python -u experiments/cs_comparison.py \
            --gpus "$GPUS" \
            --per_gpu_tasks $PER_GPU_TASKS \
            --seeds "$SEEDS" \
            --tasks "$TASKS" \
            --methods "$METHODS" \
            --model "$MODEL" \
            --lr $LR \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --weight_decay $WEIGHT_DECAY \
            --warmup_ratio $WARMUP_RATIO \
            --grad_accum $GRAD_ACCUM \
            --jelly_mode "$JELLY_MODE" \
            --switch_epoch $SWITCH_EPOCH \
            --r $R \
            --alpha $ALPHA \
            --lora_dropout $LORA_DROPOUT \
            --target_modules "$TARGET_MODULES" \
            --wandb_project "${WANDB_PROJECT}" \
            --wandb_entity "$WANDB_ENTITY" \
            --test
    else
        echo "[실험 모드]"
        python -u experiments/cs_comparison.py \
            --gpus "$GPUS" \
            --per_gpu_tasks $PER_GPU_TASKS \
            --seeds "$SEEDS" \
            --tasks "$TASKS" \
            --methods "$METHODS" \
            --model "$MODEL" \
            --lr $LR \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --weight_decay $WEIGHT_DECAY \
            --warmup_ratio $WARMUP_RATIO \
            --grad_accum $GRAD_ACCUM \
            --r $R \
            --jelly_mode "$JELLY_MODE" \
            --switch_epoch $SWITCH_EPOCH \
            --alpha $ALPHA \
            --lora_dropout $LORA_DROPOUT \
            --target_modules "$TARGET_MODULES" \
            --wandb_project "${WANDB_PROJECT}" \
            --wandb_entity "$WANDB_ENTITY"
    fi

    echo ""
    echo ">>> Switch Epoch $SWITCH_EPOCH 실험 완료"
done

echo ""
echo "============================================================"
echo " 모든 Switch Epoch 실험 완료!"
echo " Switch Epochs: $SWITCH_EPOCHS"
echo " 결과는 outputs/ 폴더에 저장됩니다."
echo "============================================================"
