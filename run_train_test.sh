task=$1    
dataset=$2    
llm_description=$3    
checkpoints_name=$4  
num_classes=$5      
cuda_device=$6
linear_name=$7



if [ ! -d ./checkpoints/${checkpoints_name} ];then
    mkdir -p ./checkpoints/${checkpoints_name}
fi

#train
CUDA_VISIBLE_DEVICES=${cuda_device} python main.py \
  -t ${task} \
  --lr 0.03  --batch-size 64 --epochs 100\
  --root ${dataset} --num-classes ${num_classes} \
  --text-load ${llm_description} \
  --checkpoints ./checkpoints/${checkpoints_name} &&

pretrained=$checkpoints_name
checkpoints_name=$linear_name &&
#evaluation_linear
./run_linear.sh ${task} ${dataset} ${pretrained} ${checkpoints_name} ${num_classes} ${cuda_device} &&


#evaluation_retrieval
./run_retrieval.sh ${task} ${dataset} ${pretrained} ${cuda_device}

