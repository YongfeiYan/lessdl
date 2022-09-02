import os
import shutil
import argparse
import torch
from itertools import chain
import copy

from simdltk.data.dataset import TranslationDataset
from simdltk.training.trainer import BasicTrainer
from simdltk.model import get_model_cls, get_arch_arch
from simdltk.model.transformer import Transformer
from simdltk import parse_args


def build_trainer(argline, rm_exp=True):
    argline = argline.split()
    args = parse_args(arglist=argline)

    train_data = TranslationDataset.build(args, 'train')
    print('train_data', train_data)
    n_lines = len(list(iter(train_data)))

    arch = get_arch_arch(args.arch)
    arch(args)
    print(args)
    model = Transformer.build(args, train_data)

    if os.path.exists(args.exp_dir) and rm_exp:
        shutil.rmtree(args.exp_dir)
    trainer = BasicTrainer(args, model, train_dataset=train_data, valid_dataset=train_data)

    return trainer, n_lines, train_data


### 测试LRStatCallback显示的问题
argline = """
    --exp-dir /tmp/exp-en-de
    --dataset translation_dataset --src-language en --tgt-language de --data-dir tests/data/mt-en-de
    --arch transformer_iwslt_de_en --dropout 0.3
    --max-batch-tokens 1000 --num-workers 0 --max-samples-in-memory 1000 --epochs 3
    --optimizer adam,lr=5e-4,weight_decay=0.0001
    --lr-scheduler inverse_sqrt,warmup_updates=2,warmup_end_lr=5e-4
    --loss label_smoothed_cross_entropy --label-smoothing 0.1
"""
trainer, n_lines, train_data = build_trainer(argline)
print(n_lines)
trainer.train()
assert (trainer.lr_scheduler.lr - 5e-4) < 0.00001, f'lr scheduler error, {trainer.lr_scheduler.lr}'


### 测试ckpt, early stopping出发后, 不能会再次训练了
argline = """
    --exp-dir /tmp/exp-en-de
    --dataset translation_dataset --src-language en --tgt-language de --data-dir tests/data/mt-en-de
    --arch transformer_iwslt_de_en --dropout 0.3
    --max-batch-tokens 12 --num-workers 0 --max-samples-in-memory 1000 --epochs 100 --earlystopping 2
    --optimizer adam,lr=5,weight_decay=0.0001
    --loss label_smoothed_cross_entropy --label-smoothing 0.1
"""

trainer, _, _ = build_trainer(argline)
trainer.train()
epochs = trainer.ckpt.epoch_counter
print('Training use', epochs, 'epochs')
trainer, _, _ = build_trainer(argline, rm_exp=False)
trainer.train()
new_epochs = trainer.ckpt.epoch_counter
assert new_epochs == epochs, f'earlystopping后, 恢复后不应该再训练, new epochs {new_epochs}, epochs {epochs}'



###
print("OK!")
