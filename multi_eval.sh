#!/usr/bin/env bash

# Script to evaluate several checkpoints (stored in /modelos)

EXPNAME=$1
SAVEDIR=/results/${EXPNAME}
MODELS=($(ls /modelos/model_0*)) # array with the paths to every model in /modelos

for model in "${MODELS[@]}"; do
    # echo ${model}
    # echo ${SAVEDIR}/eval_${model:9:13}

    CUDA_LAUNCH_BLOCKING=1 python3 main.py --num-gpus 1 --config-file configs/paco/defrcn_det_r101_base.yaml --eval-only \
    --opts MODEL.WEIGHTS ${model} \
           OUTPUT_DIR ${SAVEDIR}/eval_${model:9:13}
done