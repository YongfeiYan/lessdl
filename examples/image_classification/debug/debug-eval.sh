# data_dir=local/data/imagenet
# exp_dir=local/image/resnet50
data_dir=local/data/imagenet-test
exp_dir=local/image/debug

devices=0,1,2,3,4,5,6,7

mkdir -p $exp_dir
rm -rf $exp_dir/*
cp $0 $exp_dir  # save scripts

PYTHONPATH=. python -m lessdl \
    --exp-dir $exp_dir \
    --data-dir $data_dir --train-split val --test-split val \
    --evaluate-only True --restore-exp-dir local/image/resnet50 --evaluate-best-ckpt False\
    &> $exp_dir/run.log &

tail -f $exp_dir/run.log
disown
echo 'log:' $exp_dir/run.log 
