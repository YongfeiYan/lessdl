import torch
from torch import nn
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
import os
from os import path
import time
import datetime
import math
import numpy as np
from collections import defaultdict
import copy
import logging
from inspect import Parameter, signature

from simdltk.data.dataloader import DataLoader
from simdltk.loss import get_loss_cls
from .utils import get_optimizer, get_lr_scheduler, save_args, move_to_device
from simdltk.utils import bool_flag
from .callbacks import Checkpoint, CallbackList, LogCallback, LRStatCallback
from . import register_trainer
from .trainer import DistributedTrainer, BasicTrainer

logger = logging.getLogger()

"""
world_size: nodes
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]

main_worker
    init
    set_device
    create model/optim/dataset
    set sampler
    set loader
    epochs:
        train_epoch
             
        eval_epoch
        LR scheduler
"""

"""New import packages"""
import torch.multiprocessing as mp 
import torch.distributed as dist 
from .callbacks import Callback


class DataSamplerEpochSetter(Callback):
    def __init__(self, dist_sampler):
        self.dist_sampler = dist_sampler
        super().__init__(True)

    def on_train_epoch_begin(self, epoch, logs=None):
        self.dist_sampler.set_epoch(epoch)
        return super().on_train_epoch_begin(epoch, logs)


@register_trainer('ddp_trainer')
class DDPTrainer(BasicTrainer):
    def __init__(self, args, model, predictor=None, train_dataset=None, valid_dataset=None, callbacks: list = None, dist_url=None, dist_backend='nccl', default_tensor_type=''):
        super().__init__(args, model, predictor, train_dataset, valid_dataset, callbacks)
        devices = [x for x in args.devices.split(',') if x]
        self.distributed = len(devices) > 1
        self._rank = None
        self._world_size = len(devices)
        self._all_devices = devices
        self._dist_url = dist_url
        self._dist_backend = dist_backend
        self._info_prefix = ''
        self._train_data_sampler = None
        self.default_tensor_type = default_tensor_type
        self.batch_size = self.batch_size * len(devices) if self.distributed else self.batch_size
    
    def _prepare_building(self):
        if self._rank is not None:
            self._info_prefix = 'Rank {} - '.format(self._rank)
            self.device = torch.device('cuda:{}'.format(self._all_devices[self._rank]))
        elif len(self._all_devices) > 0:
            assert len(self._all_devices) == 1, 'Only one device is support is required in non-ddp mode, but got {}'.format(self._all_devices)
            self.device = 'cuda:{}'.format(self._all_devices[0])
        else:
            self.device = None
        if self.default_tensor_type:
            logger.info(self._info_prefix + 'set default tensor type to {}'.format(self.default_tensor_type))
            torch.set_default_tensor_type(self.default_tensor_type)
        if self.device is not None:
            torch.cuda.set_device(self.device)
            self.model.to(self.device)
        logger.info(self._info_prefix + 'Device {}'.format(self.device))
        a = torch.Tensor([0.0])
        logger.info(self._info_prefix + 'Default device: {}, default float: {}'.format(a.device, a.dtype))

    def _build_callbacks(self):
        # Add train data sampler
        if self.distributed:
            assert self._train_data_sampler is not None, 'Sampler should be set before callbacks'
            self._trainer_callbacks.append(DataSamplerEpochSetter(self._train_data_sampler))
        super()._build_callbacks()
        # Set rank for log anc checkpoint callback
        for c in self.callbacks + self._trainer_callbacks:
            c.set_rank(self._rank)
        
    def _build_sampler(self, dataset, shuffle, drop_last, is_train=True):
        if self.distributed:
            logger.warn('{} set droplast=False'.format(self._info_prefix))
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
            if is_train:
                self._train_data_sampler = sampler
            return sampler
        else:
            return None

    def _train_worker(self, rank):
        self._rank = rank
        dist.init_process_group(self._dist_backend, init_method=self._dist_url, world_size=self._world_size, rank=rank)
        self._build_trainer()
        # logger.info('Rank {}, model.device {}'.format(rank, next(self.model.parameters()).device))
        super().train()

    # def train_batch(self, batch, batch_counter=None):
    #     logger.warn('X' * 20)
    #     # return super().train_batch(batch, batch_counter)
    
    # def evaluate_batch(self, batch, predictor=None):
    #     logger.warn('X' * 20)
    #     # return super().evaluate_batch(batch, predictor)

    def train(self):
        # init training 
        if not self.distributed:
            return super().train()
        logger.info('Launch DDP training, world_size {}, devices {}'.format(self._world_size, self._all_devices))
        mp.spawn(self._train_worker, nprocs=len(self._all_devices))

    @classmethod
    def add_training_args(cls, parser, arglist=None):
        # use devices to specify gpus instead of CUDA_VISIBLE_DEVICES
        super().add_training_args(parser, arglist)
        parser.add_argument('--devices', type=str, default='', help='0,1')
        parser.add_argument('--dist-url', type=str, default='env://', help='e.g. tcp://127.0.0.1:PORT')
        parser.add_argument('--dist-backend', type=str, default='nccl')
        parser.add_argument('--default-tensor-type', type=str, default='', help='e.g. torch.cuda.FloatTensor')

    @classmethod
    def build(cls, args, model, train_data, valid_data, callbacks: list = None):
        return cls(args, model, train_data, valid_data, callbacks, dist_url=args.dist_url, dist_backend=args.dist_backend,  
            default_tensor_type=args.default_tensor_type
        )
