task=$1     
dataset=$2        
pretrained=$3     
cuda_device=$4  


if [ ! -d ./checkpoints_linear/${checkpoints_name} ];then
    mkdir -p ./checkpoints_linear/${checkpoints_name}
fi

CUDA_VISIBLE_DEVICES=${cuda_device} python retrieval.py \
  -t ${task} \
  --batch-size 256 \
  --root ${dataset} \
  --resume ./checkpoints/${pretrained}/model_last.pth
