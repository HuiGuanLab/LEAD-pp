#!/usr/bin/env bash
set -euo pipefail

task=$1
dataset=$2
llm_description=$3
checkpoints_name=$4
num_classes=$5
cuda_device=$6

if [ ! -d ./checkpoints/${checkpoints_name} ]; then
    mkdir -p ./checkpoints/${checkpoints_name}
fi

CUDA_VISIBLE_DEVICES=${cuda_device} python main.py \
  -a clip \
  --lr 0.03 \
  --batch-size 64 \
  --epochs 100 \
  --root ${dataset} \
  --crop_root ./DDT/crop_dataset/${task}_crop \
  --results-dir ./checkpoints/${checkpoints_name} \
  --num-classes ${num_classes} \
  --text-load ${llm_description}