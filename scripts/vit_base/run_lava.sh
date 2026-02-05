#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

GPUS="0,1,2,3"
PER_GPU=2
SEEDS="42"
TASKS="dtd eurosat gtsrb resisc45 sun397 svhn"

# LAVA Parameters
R=8
ALPHA=16
LAMBDA_VIB=1.0
LAMBDA_STAB=0.1
LAMBDA_LATENT=1.0

bash /home/dongwoo39/LAVA/scripts/vit_base/run_parallel.sh lava "$GPUS" $PER_GPU "$TASKS" "$SEEDS" $LAMBDA_VIB $LAMBDA_STAB $LAMBDA_LATENT $ALPHA $R
