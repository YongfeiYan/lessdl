# debug 
# data_dir=local/nerf_synthetic/test_dataset
# exp_dir=local/nerf_exp/test 
# testskip=1
# epochs=3
# train 
exp_dir=local/nerf_exp/lego_0925_first_run
data_dir=local/nerf_synthetic/lego 
testskip=8
epochs=15
devices=4

log_batches=100

mkdir -p $exp_dir

cp $0 $exp_dir

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$devices python -u examples/nerf/train.py\
    --exp-dir $exp_dir --epochs $epochs \
    --data-dir $data_dir --dataset nerf_dataset_blender --dataset-type blender --white-bkgd True \
    --use-viewdirs True --N-samples 64 --N-importance 128 --N-rand 1024 --batch-size 1024 --chunk 32768 \
    --half-res True --near 2 --far 6 --testskip $testskip --lindisp False --multires 10 --multires-views 4 \
    --model NeRFModel --netchunk 65535 --netdepth 8 --netdepth-fine 8 --netwidth 256 --netwidth-fine 256 --no-ndc False \
    --perturb 1.0 --raw-noise-std 0.0  \
    --trainer dist_trainer --devices 0 --render-every-epochs 3 --eval-every-n-epochs 3 --log-every-n-batches $log_batches\
    --optimizer adam,lr=0.0005 \
    --lr-scheduler exponential_decay_lr,decay_rate=0.1,decay_steps=500000 \
    --loss NoopLoss \
    --only-evaluate True \
    --load-ref-model /apdcephfs/share_1110423/cosiyan/nerf-pytorch/logs/lego_batch/200000.tar \
    | tee -a $exp_dir/eval.log

echo 'log:' $exp_dir/eval.log
