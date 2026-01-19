task=$1
pretrained=$2
checkpoints_name=$3
num_classes=$4
cuda_device=$5

# create checkpoint dir
if [ ! -d "./checkpoints_linear/${checkpoints_name}" ]; then
    mkdir -p "./checkpoints_linear/${checkpoints_name}"
fi

# auto root from task
root="./DDT/crop_dataset/${task}_crop"

CUDA_VISIBLE_DEVICES=${cuda_device} python linear_probing.py \
  --lr 30. --batch-size 256 --epochs 100 \
  --crop-root "${root}" \
  --num_classes ${num_classes} \
  --resume "./checkpoints/${pretrained}/model_last.pth" \
  --results_dir "./checkpoints_linear/${checkpoints_name}"
