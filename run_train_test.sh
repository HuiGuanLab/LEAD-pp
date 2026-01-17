#!/usr/bin/env bash
set -euo pipefail

task=$1
dataset=$2
llm_description=$3
train_ckpt_name=$4
num_classes=$5
cuda_device=$6
linear_name=$7

# 1) train
./run_train.sh \
  "${task}" \
  "${dataset}" \
  "${llm_description}" \
  "${train_ckpt_name}" \
  "${num_classes}" \
  "${cuda_device}"

# after train, pretrained == train_ckpt_name
pretrained="${train_ckpt_name}"

# 2) linear probing
./run_linear.sh \
  "${task}" \
  "${pretrained}" \
  "${linear_name}" \
  "${num_classes}" \
  "${cuda_device}"

# 3) retrieval
./run_retrieval.sh \
  "${task}" \
  "${pretrained}" \
  "${cuda_device}"