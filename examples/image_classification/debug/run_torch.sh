
exp_dir=local/image/torch_example-seed

work_dir=$(pwd)
mkdir -p $exp_dir
rm -rf $exp_dir/*
cp "$0" $exp_dir

cd $exp_dir
python -u $work_dir/examples/image_classification/main.py -a resnet50 --workers 32 --dist-url 'tcp://127.0.0.1:8899' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 $work_dir/local/data/imagenet &> run.log &

sleep 3
jobs 
disown -a 
