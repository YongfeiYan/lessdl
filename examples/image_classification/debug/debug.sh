
exp_dir=local/image/debug2
data_dir=local/data/imagenet-test
# data_dir=local/data/imagenet

CUDA_VISIBLE_DEVICES=0 bash examples/image_classification/run_train.sh \
    $exp_dir \
    $data_dir \
    1 0 tcp://127.0.0.1:8899 \
    resnet50 3 1 0.1
