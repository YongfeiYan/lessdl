
exp_dir=local/image/resnet502
# data_dir=local/data/imagenet-test
data_dir=local/data/imagenet

epochs=120 # 90
log_batches=100
devices=0,1,2,3,4,5,6,7
batch_size=32  # total samples on all gpus is 256

mkdir -p $exp_dir
rm -rf $exp_dir/*
cp $0 $exp_dir  # save scripts

PYTHONPATH=. python -u scripts/train.py\
    --seed 13 --exp-dir $exp_dir --epochs $epochs --batch-size $batch_size\
    --data-dir $data_dir --dataset imagenet_dataset --test-split none  --num-workers 4 \
    --model torchvision_models --model-name resnet50 \
    --trainer ddp_trainer --devices $devices --log-every-n-batches $log_batches --earlystopping 30 \
    --callbacks acc_cb --acc-cb-topk 1,5 \
    --dist-url tcp://127.0.0.1:8899 \
    --optimizer sgd,lr=0.1,momentum=0.9,weight_decay=0.0001 \
    --lr-scheduler steplr,step_size=30,gamma=0.1 \
    --loss cross_entropy \
    &> $exp_dir/run.log &

sleep 3
echo 'jobs:' 
jobs 
disown 
echo 'See log at:' $exp_dir/run.log
