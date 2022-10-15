
[English](README.en.md) | [简体中文](README.md)

# 介绍
lessdl是Simple Deep Learning Toolkit的缩写，基于PyTorch，旨在用极简的接口构建神经网络和训练，能方便地复用模型，以及修改训练流程的各个环节，主要用于复现经典模型和进行一些实验。


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


# 示例
## [Machine Translation: Transformer, IWSLT2014](examples/translation-iwslt14-en-de/README.md)
BLEU结果: 
| Model | de -> en | en -> de |
| :--:  |  :--:    |   :--:   |
| Transformer | 33.27 | 27.72 | 
| Tied Transformers | 35.10 | 29.07 | 
| fairseq | 34.54 | 28.61 | 
| Ours | 34.36 | 28.33 | 


# Reference
fairseq </br>
AllenNLP </br>
[Read TFRecord](https://github.com/vahidk/tfrecord/blob/master/tfrecord/reader.py)</br>
