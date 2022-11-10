
exp_dir=local/image/debug
data_dir=local/data/imagenet-test
# data_dir=local/data/imagenet

epochs=3 # 90
log_batches=100
devices1=0
devices2=2
batch_size=16  # total samples on all gpus is 256

mkdir -p $exp_dir
rm -rf $exp_dir/*
cp $0 $exp_dir  # save scripts

PYTHONPATH=. python -u scripts/train.py\
    --seed 19 --exp-dir $exp_dir --epochs $epochs --batch-size $batch_size\
    --data-dir $data_dir --dataset imagenet_dataset --test-split none  --num-workers 4 \
    --model torchvision_models --model-name resnet50 \
    --trainer ddp_trainer --devices $devices1 --nnodes 2 --node-rank 0 \
    --log-every-n-batches $log_batches\
    --callbacks acc_cb --acc-cb-topk 1,5 \
    --dist-url tcp://127.0.0.1:8899 \
    --optimizer sgd,lr=0.01,momentum=0.9,weight_decay=0.0001 \
    --lr-scheduler steplr,step_size=1,gamma=0.1 \
    --loss cross_entropy \
    &> $exp_dir/run1.log &

PYTHONPATH=. python -u scripts/train.py\
    --seed 19 --exp-dir $exp_dir --epochs $epochs --batch-size $batch_size\
    --data-dir $data_dir --dataset imagenet_dataset --test-split none  --num-workers 4 \
    --model torchvision_models --model-name resnet50 \
    --trainer ddp_trainer --devices $devices2 --nnodes 2 --node-rank 1 \
    --log-every-n-batches $log_batches\
    --callbacks acc_cb --acc-cb-topk 1,5 \
    --dist-url tcp://127.0.0.1:8899 \
    --optimizer sgd,lr=0.01,momentum=0.9,weight_decay=0.0001 \
    --lr-scheduler steplr,step_size=1,gamma=0.1 \
    --loss cross_entropy \
    &> $exp_dir/run2.log &

echo 'jobs:' 
tail -f $exp_dir/run*.log
jobs 
disown 
