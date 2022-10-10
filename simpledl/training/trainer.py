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
import torch.multiprocessing as mp 
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel

from simpledl.data.dataloader import DataLoader
from simpledl.loss import get_loss_cls
from .utils import get_optimizer, get_lr_scheduler, save_args, move_to_device
from simpledl.utils import bool_flag
from .callbacks import Checkpoint, CallbackList, LogCallback, LRStatCallback, Callback
from . import register_trainer


logger = logging.getLogger()


def forward_model(model, batch):
    if isinstance(model, DistributedDataParallel):
        sig = signature(model.module.forward)
    else:
        sig = signature(model.forward)
    feed = {}
    for k in sig.parameters:
        param = sig.parameters[k]
        if k in batch:
            feed[k] = batch[k]
        elif param.kind is not Parameter.POSITIONAL_OR_KEYWORD or (param.default is param.empty):
            raise RuntimeError(f'{k} is required in {model.__class__}.forward but not provided in batch {list(batch.keys())}')
    return model.forward(**feed)


@register_trainer('basictrainer')
class BasicTrainer:
    def __init__(self, args, model, predictor=None, train_dataset=None, valid_dataset=None, callbacks: list=None):
        callbacks = callbacks or []
        assert isinstance(callbacks, list)
        self.callbacks = callbacks  # For user
        self._trainer_callbacks = []  # For training procedure
        assert train_dataset is not None
        self.predictor = predictor
        self.args = args
        self.batch_size = args.batch_size
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self._ready = False
    
    def _build_trainer(self):
        """Lazy building for distributed training
        """
        if self._ready:
            return 
        self._prepare_building()  
        # self.valid_loader = self._build_dataloader(valid_dataset, is_train=False)
        self._build_optimizer()
        self._build_lr_scheduler()
        self._build_loss()
        self.train_loader = self._build_dataloader(self.train_dataset, is_train=True)
        os.makedirs(self.args.exp_dir, exist_ok=True)
        self._build_callbacks()
        self._ready = True

    def _build_callbacks(self):
        args, callbacks, model = self.args, self._trainer_callbacks, self.model 
        # TODO: add log: max norm of gradients, grad clip and norm value, tokens/s, training time etc.
        lr_cb = LRStatCallback(optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)
        callbacks.append(lr_cb)
        log_cb = LogCallback(self.callbacks + callbacks, 
            log_every_n_batches=args.log_every_n_batches,
            log_every_n_epochs=args.log_every_n_epochs
        )
        # 总是将log callback, checkpoint callback放到最后, 打印的时候, 确保前面的callback已经计算了
        callbacks.append(log_cb)
        self.log_cb = log_cb
        self.ckpt = Checkpoint(args.exp_dir, self.callbacks + callbacks, last_to_keep=args.last_to_keep, earlystopping=args.earlystopping)
        # self.callbacks = callbacks
        # set model for all
        self.ckpt.set_model(self.model)
        for c in self.callbacks + callbacks:
            c.set_model(model)

    def _prepare_building(self):
        # 只考虑CPU和一个GPU的情况
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = None
        logger.info('Device {}'.format(self.device))
        if self.device:
            self.model.to(self.device)

    def _build_sampler(self, dataset, shuffle, drop_last):
        return None

    def _build_dataloader(self, dataset, is_train=True):
        """
        TODO: 分不同类型的数据, 构建不同loader class
        """
        shuffle = True if is_train and self.args.shuffle else False
        drop_last = True if is_train and shuffle else False
        sampler = self._build_sampler(dataset, shuffle, drop_last, is_train)
        if self.args.max_batch_tokens and self.batch_size:
            logger.warning(f'max_batch_tokens is specified, overwrite batch_size')
            batch_size = 0
        else:
            batch_size = self.batch_size
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, max_batch_tokens=self.args.max_batch_tokens,
            max_samples_in_memory=self.args.max_samples_in_memory, sort_key=self.args.sort_key if is_train else None,
            num_workers=self.args.num_workers, sampler=sampler, drop_last=drop_last, pin_memory=self.args.pin_memory,
        )
        return dl

    def _build_optimizer(self):
        params = self.model.parameters()
        self.optimizer = get_optimizer(params, self.args.optimizer)
        self.grad_clip = self.args.grad_clip
        self.grad_norm = self.args.grad_norm

    def _build_loss(self):
        loss_cls = get_loss_cls(self.args.loss)
        self.loss = loss_cls.build(self.args, self.model, self.train_dataset)

    def _build_lr_scheduler(self):
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.args.lr_scheduler)
    
    def forward_model(self, model, batch):
        return forward_model(model, batch)

    def train(self):
        self._build_trainer()
        save_args(self.args, os.path.join(self.args.exp_dir, 'kwargs.json'))
        cb_list = CallbackList(self.callbacks + self._trainer_callbacks + [self.ckpt])  # checkpoint only at training
        last = self.ckpt.restore(last=True) or -1
        if last == -1:
            logger.info('Begin to train')
            cb_list.on_train_begin({})
        else:
            logger.info(f'continue to train from epoch {last + 1}, total epochs {self.args.epochs}')
        batch_counter = self.ckpt.batch_counter
        for e in range(last + 1, self.args.epochs):
            logger.info(f'Epoch {e + 1}/{self.args.epochs} ----------------------------')
            if cb_list.should_stop_training():
                break
            cb_list.on_train_epoch_begin(e)
            self.model.train()
            for batch in self.train_loader:            
                if cb_list.should_stop_training():
                    break
                cb_list.on_train_batch_begin({})
                out = self.train_batch(batch, batch_counter)
                cb_list.on_train_batch_end(batch, out)
                batch_counter += 1
                if self.args.eval_every_n_batches and batch_counter % self.args.eval_every_n_batches == 0:
                    self._evaluate(self.valid_dataset, cb_list, self.predictor)
                    self.model.train()
            cb_list.on_train_epoch_end(e, {})
            if self.args.eval_every_n_epochs and (e + 1) % self.args.eval_every_n_epochs == 0:
                self._evaluate(self.valid_dataset, cb_list, self.predictor)
        if last < self.args.epochs - 1 and e != self.args.epochs - 1:
            logger.info(f'Early stopped.')
        logger.info(f'Best evaluation {self.ckpt.cmp_key}: {self.ckpt.best_metric}')
        cb_list.on_train_end({})

    def move_to_device(self, batch):
        """
        TODO non-block for pin_memory=True
        """
        if self.device:
            batch = move_to_device(batch, self.device)
        return batch

    def train_batch(self, batch, batch_counter=None):
        """
        Return a dict
        """
        # self.model.train()
        assert self.model.training == True
        self.optimizer.zero_grad()
        batch = self.move_to_device(batch)
        out = self.forward_model(self.model, batch)
        loss_dict = self.loss(batch, out)
        out.update(loss_dict)
        if torch.isnan(out['loss']).any():
            logger.warn('NaN in loss, NaN count is {}'.format(torch.sum(torch.isnan(out['loss'])).item()))
            logger.warn('Skip batch because of NaN')
            return out
        out['loss'].backward()

        if self.grad_clip:
            clip_grad_value_(self.model.parameters(), self.grad_clip)
        if self.grad_norm:
            clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step_batch(batch_counter)

        return out

    def evaluate_batch(self, batch, predictor=None):
        # TODO 使用predictor
        # self.model.eval()
        # with torch.no_grad():  # move to _evaluate
        assert self.model.training == False
        batch = self.move_to_device(batch)
        if predictor is None:
            out = self.forward_model(self.model, batch)
            loss_dict = self.loss(batch, out)
            out.update(loss_dict)
        return out

    def _evaluate(self, dataset, callbacks=None, predictor=None):
        """
        TODO: 也可以使用predictor, 比如在测试的时候, 用到beam search测试bleu score
        Return a dict, measuring something.
        """
        # 当训练的过程中测试的时候, 需要保存ckpt, 当预测eval的时候, 不需要保存ckpt
        """
        valid:
        - 对每个batch, 进行predict和loss等, 将结果加入到callback
        - epoch结束, 进行结果整合
        - 根据结果等好坏, checkpoint进行保存. 办法: 区分eval还是test, eval的时候, 用到ckpt
        test:
        - 同valid 1
        - 同valid 2
        - 要返回结果  办法: 添加类似logcallback的类, 只不过是收集结果.
        办法: 除了checkpoint等情况, 其余都一样, 因此valid和test用一个模式, 只不过checkpoint要区分是否加入到callbacks里
        """
        callbacks = callbacks or []
        if isinstance(callbacks, list):
            callbacks = CallbackList(callbacks)
        callbacks.set_model(self.model)
        dataloader = self._build_dataloader(dataset, is_train=False)
        self.model.eval()
        with torch.no_grad():
            logger.info('Begin Evaluation')
            callbacks.on_evaluate_begin()
            for batch in dataloader:
                callbacks.on_evaluate_batch_begin(batch)
                out = self.evaluate_batch(batch, predictor)
                callbacks.on_evaluate_batch_end(batch, out)
            callbacks.on_evaluate_end()
            logger.info('End Evaluation')
        status = callbacks.get_evaluate_status()
        return status

    def evaluate(self, dataset, callbacks=None, predictor=None):
        self._build_trainer()
        if callbacks is None:
            callbacks = self.callbacks + self._trainer_callbacks
        else:
            callbacks = callbacks + [self.log_cb]
        predictor = predictor or self.predictor
        return self._evaluate(dataset, callbacks, predictor)

    @classmethod
    def add_optimizer_args(cls, parser, arglist=None):
        parser.add_argument('--optimizer', type=str, default='adam',
            help='optim,lr=0.001,delta=0.00001....'
        )
        parser.add_argument('--grad-clip', type=float, default=None)
        parser.add_argument('--grad-norm', type=float, default=None)

    @classmethod
    def add_lr_scheduler_args(cls, parser, arglist=None):
        parser.add_argument('--lr-scheduler', type=str, default='none',
            help='和adam的格式相同, 比如 inverse_sqrt,warmup_updates=4000,warmup_end_lr=5e-4'
        )

    @classmethod
    def add_loss_args(cls, parser, arglist=None):
        parser.add_argument('--loss', type=str, default='cross_entropy')
        # TODO 解析loss, 然后把对应loss的args加上去.
        args, _ = parser.parse_known_args(arglist)
        loss_cls = get_loss_cls(args.loss)
        loss_cls.add_args(parser, arglist)

    @classmethod
    def add_dataloader_args(cls, parser, arglist=None):
        parser.add_argument('--batch-size', type=int, metavar='N', required=True)
        parser.add_argument('--max-batch-tokens', type=int, default=0, help='e.g. 4096')
        parser.add_argument('--shuffle', type=bool_flag, default=True, help='Shuffle training data')
        parser.add_argument('--num-workers', type=int, default=0)
        parser.add_argument('--max-samples-in-memory', type=int, default=0)
        parser.add_argument('--pin-memory', type=bool_flag, default=False)
        # TODO: 检查是否可以, sort_key=args.sort_key
        parser.add_argument('--sort-key', type=str, default='_size')  # default='src_len')  #  根据dataset的哪个字段进行sort
        # parser.add_argument('--batch-max-tokens', type=int,
        #     help='可以通过指定max-tokens间接指定batch size')

    @classmethod
    def add_training_args(cls, parser, arglist=None):
        parser.add_argument('--exp-dir', type=str, required=True, help='Experiment dir')
        parser.add_argument('-e', '--epochs', type=int)
        parser.add_argument('--earlystopping', type=int, default=10)
        parser.add_argument('--last-to-keep', type=int, default=1)
        parser.add_argument('--eval-every-n-epochs', type=int, default=1)
        parser.add_argument('--eval-every-n-batches', type=int)
        parser.add_argument('--log-every-n-batches', type=int)
        parser.add_argument('--log-every-n-epochs', type=int, default=1)

    @classmethod
    def add_args(cls, parser, arglist=None):
        cls.add_dataloader_args(parser, arglist)
        cls.add_loss_args(parser, arglist)
        cls.add_lr_scheduler_args(parser, arglist)
        cls.add_optimizer_args(parser, arglist)
        cls.add_training_args(parser, arglist)

    @classmethod
    def build(cls, args, model, train_data, valid_data, callbacks: list = None):
        return cls(args, model, train_dataset=train_data, valid_dataset=valid_data, callbacks=callbacks)


# @register_trainer('dist_trainer')
# class DistributedTrainer(BasicTrainer):
#     def __init__(self, args, model, predictor=None, train_dataset=None, valid_dataset=None, callbacks: list = None):
#         super().__init__(args, model, predictor, train_dataset, valid_dataset, callbacks)

#     def _prepare_building(self):
#         devices = [d for d in self.args.devices.split(',') if d]
#         if len(devices) > 0:
#             assert len(devices) == 1
#             self.device = 'cuda:{}'.format(devices[0])
#             self.model.to(self.device)
#         else:
#             self.device = None
#         logger.info('Device {}'.format(self.device))

#     @classmethod
#     def add_training_args(cls, parser, arglist=None):
#         # use devices to specify gpus instead of CUDA_VISIBLE_DEVICES
#         super().add_training_args(parser, arglist)
#         parser.add_argument('--devices', type=str, default='', help='0,1')
#         parser.add_argument('--dist-url', type=str, default='')
#         parser.add_argument('--dist-backend', type=str, default='nccl')


class DataSamplerEpochSetter(Callback):
    def __init__(self, dist_sampler):
        self.dist_sampler = dist_sampler
        super().__init__(True)

    def on_train_epoch_begin(self, epoch, logs=None):
        self.dist_sampler.set_epoch(epoch)
        return super().on_train_epoch_begin(epoch, logs)


class DistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """Support set default tensor type
    """
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            device = torch.tensor(0.).device
            g = torch.Generator(device=device)  
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


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
        # self.batch_size = self.batch_size * len(devices) if self.distributed else self.batch_size
    
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
            if self.distributed:
                gpu = int(self._all_devices[self._rank])
                logger.info(self._info_prefix + 'use gpu {}'.format(gpu))
                self.model = DistributedDataParallel(self.model, device_ids=[gpu])
        logger.info(self._info_prefix + 'Device {}'.format(self.device))
        a = torch.Tensor([0.0])
        logger.info(self._info_prefix + 'Default device: {}, default float: {}'.format(a.device, a.dtype))

    def _build_callbacks(self):
        # Add train data sampler
        if self.distributed:
            assert self._train_data_sampler is not None, 'Sampler should be set before callbacks'
            self._trainer_callbacks.append(DataSamplerEpochSetter(self._train_data_sampler))
        # NOTE: call in BasicTrainer the setup log_callback and checkpoint checkpoint, which depends on
        # callbacks and _trianer_callback
        super()._build_callbacks()  
        # Set rank for log and checkpoint callback
        for c in self.callbacks + self._trainer_callbacks:
            c.set_rank(self._rank)
        self.ckpt.set_rank(self._rank)

    def _build_sampler(self, dataset, shuffle, drop_last, is_train=True):
        if self.distributed:
            logger.warn('{} set droplast=False'.format(self._info_prefix))
            sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
            if is_train:
                self._train_data_sampler = sampler
            return sampler
        else:
            return None

    def _train_worker(self, rank):
        self._rank = rank
        dist.init_process_group(self._dist_backend, init_method=self._dist_url, world_size=self._world_size, rank=rank)
        self._build_trainer()
        super().train()

    def train(self):
        if not self.distributed:
            return super().train()
        assert self._dist_url, 'dist_url should be like: --dist-url tcp://127.0.0.1:PORT'
        logger.info('Launch DDP training, world_size {}, devices {}'.format(self._world_size, self._all_devices))
        mp.spawn(self._train_worker, nprocs=len(self._all_devices))

    @classmethod
    def add_training_args(cls, parser, arglist=None):
        # use devices to specify gpus instead of CUDA_VISIBLE_DEVICES
        super().add_training_args(parser, arglist)
        parser.add_argument('--devices', type=str, default='', help='0,1')
        parser.add_argument('--dist-url', type=str, default='', help='e.g. tcp://127.0.0.1:PORT')
        parser.add_argument('--dist-backend', type=str, default='nccl')
        parser.add_argument('--default-tensor-type', type=str, default='', help='e.g. torch.cuda.FloatTensor')

    @classmethod
    def build(cls, args, model, train_data, valid_data, callbacks: list = None):
        return cls(args, model, train_dataset=train_data, valid_dataset=valid_data, callbacks=callbacks, dist_url=args.dist_url, dist_backend=args.dist_backend,  
            default_tensor_type=args.default_tensor_type
        )
