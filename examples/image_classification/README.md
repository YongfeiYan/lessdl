# Image Classification

先检查训练过程各个指标是否对齐了，比如数据量/loss等指标大小/lr变化/训练速度等等
如何快速debug：数据 / 模型 / 优化 / metrics
数据 + 模型 + metrics ：用参考模型和数据进行计算
优化 ：固定seed比对输出
快速：减少数据量，避免全量训练

TODO 整理模型比对和grad比对的函数, 便于以后使用; 测试单卡的运行


## Data
### ImageNet
```bash 
# download train, valid data 
data_dir=local/data
mkdir -p $data_dir
cp examples/image_classification/extract_ILSVRC.sh $data_dir
cd $data_dir
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar
# extract data
bash extract_ILSVRC.sh
```

### MNIST

### CIFAR


## Training
### ImageNet
```bash 
# run, TODO sync_batch_norm / DistributedOptimizer / pin_memory of dataloader 
bash examples/image_classification/train.sh  
python 
# PyTorch ImageNet Example, 8 gpus 

# pyTorch example, eval, 8 gpus 
python -u examples/image_classification/main_eval.py -a resnet50 --workers 32 --dist-url 'tcp://127.0.0.1:8899' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 local/data/imagenet --evaluate --resume local/image/torch_examples/model_best.pth.tar &> local/image/torch_examples/eval.log 

# ImageNet ref
PYTHONPATH=. python -u main.py local/data/imagenet -a resnet50 --lr 0.01
```


## Results

alexnet        x
densenet161  acc1:78.549995, acc5:94.229996 30min  32x8
vgg19          x
vgg19_bn      run
renset18      run
resnet34     run 
resnet50     acc1:76.466003, acc5:93.173996  22min-25min/epoch  32x8 
resnet101    run 
resnet152    acc1:78.447998, acc5:94.088013  27min  32x8


## Reference 
- [PyTorch ImageNet Example](https://github.com/pytorch/examples/tree/main/imagenet)
- [ImageNet related models](https://github.com/jiweibo/ImageNet)
- [Paperwithcode ImageNet](https://paperswithcode.com/dataset/imagenet)
- [HuggingFace ImageNet 1k](https://huggingface.co/datasets/imagenet-1k)




