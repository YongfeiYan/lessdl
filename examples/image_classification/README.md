# Image Classification

TODO 整理模型比对和grad比对的函数, 便于以后使用; 测试单卡的运行
TODO En README / pip / readthedocs / add import-lib

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

alexnet       run
densenet161  acc1:78.020004, acc5:94.042007 90  0.1
vgg19         run
vgg19_bn      run
renset18      run
resnet34     acc1:73.148003, acc5:91.350006  90 0.1 
resnet50     acc1:75.802002, acc5:92.82  90  0.1 
resnet101    acc1:77.386002, acc5:93.663994  90   0.1 
resnet152    acc1:77.989998, acc5:93.877998  90  0.1 

densenet161  acc1:78.549995, acc5:94.229996 120  0.1
resnet34     acc1:73.903999, acc5:91.669998  120 0.1
resnet50     acc1:76.466003, acc5:93.173996  120  0.1 
resnet152    acc1:78.447998, acc5:94.088013  120  0.1 

## Reference 
- [PyTorch ImageNet Example](https://github.com/pytorch/examples/tree/main/imagenet)
- [ImageNet related models](https://github.com/jiweibo/ImageNet)
- [Paperwithcode ImageNet](https://paperswithcode.com/dataset/imagenet)
- [HuggingFace ImageNet 1k](https://huggingface.co/datasets/imagenet-1k)
