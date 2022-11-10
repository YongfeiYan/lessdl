
exp_dir=local/image/torch_example-debug

work_dir=$(pwd)
mkdir -p $exp_dir
rm -rf $exp_dir/*
cp "$0" $exp_dir

cd $exp_dir
PYTHONPATH=$work_dir python -u $work_dir/examples/image_classification/main_eval_debug.py --seed 13 -a resnet50 --workers 32 --dist-url 'tcp://127.0.0.1:8898' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 $work_dir/local/data/imagenet &> run.log &

sleep 3
jobs 
disown -a 
tail -f run.log
