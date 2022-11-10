
[English](README.en.md) | [简体中文](README.md)

# 介绍
lessdl是Less Deep Learning Toolkit的缩写，基于PyTorch，旨在用极简的接口构建神经网络和训练，能方便地复用模型，以及修改训练流程的各个环节，主要用于复现经典模型和进行一些实验。


# 关键接口
## Data
不同任务的数据格式和处理方法都不相同，如果对数据进行抽象处理的话，会使整个代码变得极其复杂，lessdl没有针对数据抽象单独的类，而是复用PyTorch的Dataset，只要求返回的batch数据是dict格式，例如
```python
batch = {
    'src': src_seq,
    'src_size': src_size, 
    'tgt': tgt_seq
}
```
在lessdl/data中提供了一些常用的数据处理类，可以方便地复用，比如TranslationDataset，根据两种语言的文本文件，构建一个Dataset。


## Model
Model的接口是forward函数，表示模型的前向计算，函数的参数可以是batch的key，或者是batch，输出也要求是一个dict，比如
```python
def forward(self, src, src_size, tgt):
    # do something 
    output_logits = calc_output_logits()
    return {
        'logits': output_logits,
    }
```

## Loss
Loss的接口是数据的一个batch和模型的前向计算的输出，输出是dict，要求有key为loss，便于后面进行梯度更新 比如
```python
def forward(self, batch, out):
    return {
        'loss': loss_calc(batch, out),
    }
```

## Optimizer
使用PyTorch的Optimizer，进行简单的包装，可以根据参数str构建需要的optimizer


## Trainer
Trainer是主要的类，包括构建数据集、Model、Loss、Optimizer、callbacks。callbacks是训练过程中使用的，用于统计每一步训练的结果、保存断点、模型评估、earlystopping等操作。

训练的流程如下:
```python
def train(...):
    prepare_callbacks()
    restore_training_if_necessary()
    for e in training_epochs:
        call_callbacks_epoch_begin()
        for batch in dataset:
            call_callbacks_batch_begin()
            model_out = forward_model(batch)  # 根据batch，调用Model的forward函数
            loss = calc_loss(model_out, batch)
            gradient_update_step()  # 进行优化
            call_callbacks_batch_end()  # 统计结果、保存断点、调整学习率等
        call_callbacks_epoch_end()    
        evaluate_if_necessary()
```
通过修改trainer，可以方便地修改数据、模型、以及训练的每个环节


# Examples
## Machine Translation 
__Models__
- Transformer </br>
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

__Results__</br>
BLEU scores on IWSLT2014: 
| Model | de -> en | en -> de |
| :--:  |  :--:    |   :--:   |
| Transformer | 33.27 | 27.72 | 
| Tied Transformers | 35.10 | 29.07 | 
| fairseq | 34.54 | 28.61 | 
| Ours | 34.36 | 28.33 | 

[Here are training details](examples/translation-iwslt14-en-de/README.md)


## Image Classification 
__Models__
- AlexNet </br>
    [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- DenseNet </br>
    [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
- VGG </br>
    [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- ResNet </br>
    [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

__Results__ </br>
Top 1 and top 5 accuracy of validation on ImageNet ILSVRC2012:
|   Model     | Acc@1     | Acc@5     | Epochs   | Learning Rate |
|   :--:      | :--:      |   :--:    |   :--:   |   :--:        |
| AlexNet     | 56.272    | 78.982    |   90     |  0.01         |
| DenseNet161 | 78.020    | 94.042    |   90     |  0.1          |
| VGG19       | 72.269    | 90.940    |   90     |  0.01         | 
| VGG19_bn    | 74.251    | 92.160    |   90     |  0.1          |
| ResNet18    | 69.753    | 89.138    |   90     |  0.1          |
| ResNet34    | 73.148    | 91.350    |   90     |  0.1          |
| ResNet50    | 75.802    | 92.820    |   90     |  0.1          | 
| ResNet101   | 77.386    | 93.664    |   90     |  0.1          |
| ResNet152   | 77.989    | 93.878    |   90     |  0.1          | 

[Here are training details](examples/image_classification/README.md)


# Reference
- [fairseq](https://github.com/facebookresearch/fairseq)
- [AllenNLP](https://allenai.org/allennlp)
- [Keras](https://keras.io/)
