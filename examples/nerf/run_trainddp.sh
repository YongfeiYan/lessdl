# debug 
data_dir=local/nerf_synthetic/test_dataset
exp_dir=local/nerf_exp/test 
testskip=1
epochs=3
# train 
# exp_dir=local/nerf_exp/lego_0927_rerun
# data_dir=local/nerf_synthetic/lego 
# testskip=8
# epochs=15

devices=0,1,2,3
batch_size=$[1024*4]

log_batches=100

mkdir -p $exp_dir
rm -rf $exp_dir/*
cp $0 $exp_dir

# CUDA_VISIBLE_DEVICES=$devices
PYTHONPATH=.  python examples/nerf/trainddp.py\
    --seed 19 --exp-dir $exp_dir --epochs $epochs \
    --data-dir $data_dir --dataset nerf_dataset_blender --dataset-type blender --white-bkgd True \
    --use-viewdirs True --N-samples 64 --N-importance 128 --N-rand 1024 --batch-size $batch_size --chunk 32768 \
    --half-res True --near 2 --far 6 --testskip $testskip --lindisp False --multires 10 --multires-views 4 \
    --model NeRFModel --netchunk 65535 --netdepth 8 --netdepth-fine 8 --netwidth 256 --netwidth-fine 256 --no-ndc False \
    --perturb 1.0 --raw-noise-std 0.0  \
    --trainer ddp_trainer --devices $devices --render-every-epochs 3 --eval-every-n-epochs 3 --log-every-n-batches $log_batches\
    --optimizer adam,lr=0.0005 \
    --lr-scheduler exponential_decay_lr,decay_rate=0.1,decay_steps=500000 \
    --loss NoopLoss &> $exp_dir/run.log &

sleep 3
echo 'jobs:' 
jobs 
disown 
echo 'log:' $exp_dir/run.log
tail -f $exp_dir/run.log








