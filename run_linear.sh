task=$1   
dataset=$2 
pretrained=$3 
checkpoints_name=$4
num_classes=$5 
cuda_device=$6 


if [ ! -d ./checkpoints_linear/${checkpoints_name} ];then
    mkdir -p ./checkpoints_linear/${checkpoints_name}
fi

CUDA_VISIBLE_DEVICES=${cuda_device} python linear_probing.py \
  -t ${task} \
  --lr 30.  --batch-size 256 --epochs 100\
  --root ${dataset} --num-classes ${num_classes} \
  --resume ./checkpoints/${pretrained}/model_last.pth\
  --checkpoints ./checkpoints_linear/${checkpoints_name}
