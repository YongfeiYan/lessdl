# Image Classification
This is a example of using lessdl to train image classification models. 
Currently, it uses ImageNet ILSVRC2012 dataset from [PyTorch ImageNet Example](https://github.com/pytorch/examples/tree/main/imagenet) and the models used are from `torchvision.models`.
The default learning rate scheduler is StepLR with initial lr=0.1 and it decays by a factor of 10 every 30 epochs.
In the future, more datasets such as CIFAR and MNIST will be supported.

## Data
### ImageNet ILSVRC2012
Download ImageNet dataset and extract it into _local/data_ dir: 
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
After extraction, _local/data_ should be like: 
```bash 
imagenet
├── train
└── val
```

### MNIST
### CIFAR


## Training
### ImageNet ILSVRC2012
```bash 
# Run on a single GPU
CUDA_VISIBLE_DEVICES=0 bash examples/image_classification/run_train.sh \
    local/exp/resnet50 \
    local/data/imagenet \
    1 0 tcp://127.0.0.1:8899 \
    resnet50 90 30 0.1
# Run on all GPUs in a single node. To train on multiple nodes, run the following cmd in each node and specify and dist_url and ranks. 
# args of run_train.sh: exp_dir data_dir nnodes node_rank dist_url model_name epochs step_size lr
bash examples/image_classification/run_train.sh \
    local/exp/resnet50 \
    local/data/imagenet \
    1 0 tcp://127.0.0.1:8899 \
    resnet50 90 30 0.1

# Run PyTorch example with 8 GPUs, https://github.com/pytorch/examples/tree/main/imagenet
python -u examples/image_classification/main_eval.py -a resnet50 --workers 32 --dist-url 'tcp://127.0.0.1:8899' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 local/data/imagenet &> torch_examples_eval.log 

# Run training script of ImageNet: https://github.com/jiweibo/ImageNet
PYTHONPATH=. python -u main.py local/data/imagenet -a resnet50 --lr 0.01
```


## Results
Top 1 and Top 5 accuracy on ImageNet ILSVRC2012 of different models:
|   Model     | Acc@1     | Acc@5     | Epochs   | Learning Rate |
|   :--:      | :--:      |   :--:    |   :--:   |   :--:        |
| AlexNet     | 56.271999 | 78.982002 |   90     |  0.01         |
| DenseNet161 | 78.020004 | 94.042007 |   90     |  0.1          |
| VGG19       | 72.269997 | 90.940002 |   90     |  0.01         | 
| VGG19_bn    | 74.251999 | 92.159996 |   90     |  0.1          |
| ResNet18    | 69.753998 | 89.138    |   90     |  0.1          |
| ResNet34    | 73.148003 | 91.350006 |   90     |  0.1          |
| ResNet50    | 75.802002 | 92.82     |   90     |  0.1          | 
| ResNet101   | 77.386002 | 93.663994 |   90     |  0.1          |
| ResNet152   | 77.989998 | 93.877998 |   90     |  0.1          | 


## Reference 
- [PyTorch ImageNet Example](https://github.com/pytorch/examples/tree/main/imagenet)
- [ImageNet related models](https://github.com/jiweibo/ImageNet)
- [Paperwithcode ImageNet](https://paperswithcode.com/dataset/imagenet)
- [HuggingFace ImageNet 1k](https://huggingface.co/datasets/imagenet-1k)
