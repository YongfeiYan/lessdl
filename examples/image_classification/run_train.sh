
if [ $# -ne 9 ]; then 
    echo 'args: exp_dir data_dir nnodes node_rank dist_url model_name epochs step_size lr'
    exit 1 
fi
exp_dir=$1
data_dir=$2
nnodes=$3
node_rank=$4 
dist_url=$5
# e.g. resnet50
model_name=$6
epochs=$7
step_size=$8
lr=$9
# exp_dir=local/image/debug
# data_dir=local/data/imagenet-test
log_batches=100
batch_size=32

mkdir -p $exp_dir
rm -rf $exp_dir/*
cp $0 $exp_dir/run_script$node_rank.sh  # save scripts
cd $(dirname "$0")/../..  # 
echo 'Files:'
ls -l 
echo 'Begin to run ...'

PYTHONPATH=. python -u scripts/train.py\
    --seed 13 --exp-dir $exp_dir --epochs $epochs --batch-size $batch_size\
    --data-dir $data_dir --dataset imagenet_dataset --test-split none  --num-workers 4 \
    --model torchvision_models --model-name $model_name \
    --trainer ddp_trainer --nnodes ${nnodes} --node-rank ${node_rank} \
    --log-every-n-batches $log_batches --earlystopping 30 \
    --callbacks acc_cb --acc-cb-topk 1,5 \
    --dist-url $dist_url \
    --optimizer sgd,lr=$lr,momentum=0.9,weight_decay=0.0001 \
    --lr-scheduler steplr,step_size=$step_size,gamma=0.1 \
    --loss cross_entropy \
    &> $exp_dir/run${node_rank}.log

echo 'Finished'

