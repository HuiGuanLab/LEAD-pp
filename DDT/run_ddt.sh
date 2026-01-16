#!/usr/bin/env bash
set -euo pipefail

# ==========================
# Usage:
#   sh run_ddt_and_crop.sh task dataset pretrained cuda_device [output_crop_root]
#
# Example:
#   sh run_ddt_and_crop.sh \
#     bird \
#     /media/data1/zzh/EAD-FFAB/EAD/bird \
#     ../resnet50-19c8e357.pth \
#     0 \
#     /media/data1/zzh/LEAD++/DDT/output_dataset
# ==========================

task=$1
dataset=$2
pretrained=$3
cuda_device=$4
output_crop_root="${5:-crop_dataset}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Python scripts
DDT_FIT_SCRIPT="main.py"
CROP_SCRIPT="get_crop.py"

# --------------------------
# Paths
# --------------------------
VEC_PATH="./DDT_result/${task}/vec_res50_${task}.npy"
MEAN_TENSOR_PATH="./DDT_result/${task}/mean_tensor_${task}.pth"
CROP_OUT="${output_crop_root}/${task}_crop"

mkdir -p "$(dirname "$VEC_PATH")"
mkdir -p "$(dirname "$MEAN_TENSOR_PATH")"
mkdir -p "$CROP_OUT"

export CUDA_VISIBLE_DEVICES=${cuda_device}

echo "============================================================"
echo "[Step 1/2] DDT fit (ImageNet ResNet50 only)"
echo "============================================================"

cmd1=(
  ${PYTHON_BIN} ${DDT_FIT_SCRIPT}
  --seed 123
  --root "${dataset}"
  --pretrained "${pretrained}"
  --batch-size 2
  --num_workers 1
  --save-vec "${VEC_PATH}"
  --save-mean-tensor "${MEAN_TENSOR_PATH}"
)

echo "[Run] ${cmd1[*]}"
"${cmd1[@]}"

if [ ! -f "${VEC_PATH}" ] || [ ! -f "${MEAN_TENSOR_PATH}" ]; then
  echo "[ERROR] DDT outputs not found"
  echo "  vec         : ${VEC_PATH}"
  echo "  mean_tensor : ${MEAN_TENSOR_PATH}"
  exit 1
fi

echo "============================================================"
echo "[Step 2/2] DDT crop ImageFolder (square output, robust)"
echo "============================================================"

# ✅ 你的 get_crop.py 新版只支持下面这些参数
cmd2=(
  ${PYTHON_BIN} ${CROP_SCRIPT}
  --seed 1234
  --root "${dataset}"
  --trans-vec "${VEC_PATH}"
  --descriptors-mean-tensor "${MEAN_TENSOR_PATH}"
  --pretrain-model resnet50
  --output "${CROP_OUT}"
)

echo "[Run] ${cmd2[*]}"
"${cmd2[@]}"

echo "============================================================"
echo "[Done]"
echo "  vec         : ${VEC_PATH}"
echo "  mean_tensor : ${MEAN_TENSOR_PATH}"
echo "  cropped dir : ${CROP_OUT}"
echo "============================================================"
