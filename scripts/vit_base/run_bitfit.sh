#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

GPUS="0,1,2,3"
PER_GPU=2
SEEDS="42"
TASKS="dtd eurosat gtsrb resisc45 sun397 svhn"

bash /home/dongwoo39/LAVA/scripts/vit_base/run_parallel.sh bitfit "$GPUS" $PER_GPU "$TASKS" "$SEEDS" 0 0 0 16 8
