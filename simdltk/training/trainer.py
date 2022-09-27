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


logger = logging.getLogger()


def forward_model(model, batch):
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
        train_dataset = self.args, self.train_dataset, self.callbacks 
        self._prepare_building()  
        # self.valid_loader = self._build_dataloader(valid_dataset, is_train=False)
        self._build_optimizer()
        self._build_lr_scheduler()
        self._build_loss()
        self._build_callbacks()
        self._ready = True
        self.train_loader = self._build_dataloader(train_dataset, is_train=True)
        os.makedirs(self.args.exp_dir, exist_ok=True)
    
    def _build_callbacks(self):
        args, callbacks, model = self.args, self._trainer_callbacks, self.model 
        # TODO: add log: max norm of gradients, grad clip and norm value, tokens/s, training time etc.
        lr_cb = LRStatCallback(optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)
        callbacks.append(lr_cb)
        log_cb = LogCallback(callbacks, 
            log_every_n_batches=args.log_every_n_batches,
            log_every_n_epochs=args.log_every_n_epochs
        )
        # 总是将log callback, checkpoint callback放到最后, 打印的时候, 确保前面的callback已经计算了
        callbacks.append(log_cb)
        self.log_cb = log_cb
        self.ckpt = Checkpoint(args.exp_dir, callbacks, last_to_keep=args.last_to_keep, earlystopping=args.earlystopping)
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
        cb_list = CallbackList(self.callbacks + [self.ckpt])  # checkpoint only at training
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
            # TODO delete 
            print('ERROR')
            train_data = defaultdict(list)

            for batch in self.train_loader:            
                if cb_list.should_stop_training():
                    break
                cb_list.on_train_batch_begin({})
                # TODO delete 
                for k, v in batch.items():
                    train_data[k].append(v)

                out = self.train_batch(batch, batch_counter)
                cb_list.on_train_batch_end(batch, out)
                batch_counter += 1
                if self.args.eval_every_n_batches and batch_counter % self.args.eval_every_n_batches == 0:
                    self._evaluate(self.valid_dataset, cb_list, self.predictor)
                    self.model.train()
            cb_list.on_train_epoch_end(e, {})
            if self.args.eval_every_n_epochs and (e + 1) % self.args.eval_every_n_epochs == 0:
                self._evaluate(self.valid_dataset, cb_list, self.predictor)
            # TODO delete 
            r = {}
            for k, v in train_data.items():
                v = torch.cat(v, 0)
                r[k] = v
            torch.save(r, '/tmp/e{}.pt'.format(e))
            
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
        # # TODO delete 
        # print('index', batch['index'].tolist())
        # print('1s ratio:', (batch['target'] == 1).float().sum().item() / batch['target'].numel())
        # print('before parameters:')
        # before_para = list(p.clone() for p in self.model.parameters())
        # for p in before_para:
        #     print(p.shape, p.requires_grad, p.grad is not None, None if p.grad is None else p.grad.sum().item())

        batch = self.move_to_device(batch)
        out = self.forward_model(self.model, batch)
        loss_dict = self.loss(batch, out)
        out.update(loss_dict)
        # # TODO delete 
        # for k, v in out.items():
        #     print(k, v.shape, v.sum().item())
        
        if torch.isnan(out['loss']).any():
            logger.warn('NaN in loss, NaN count is {}'.format(torch.sum(torch.isnan(out['loss'])).item()))
            logger.warn('Skip batch because of NaN')
            return out
        out['loss'].backward()
        # # TODO delete 
        # print('after backward')
        # for p in self.model.parameters():
        #     print(p.shape, 'required_grad', p.requires_grad, 'grad != None', p.grad is not None, ', sum grad:', p.grad.abs().sum().item())
        # print('model.parateters')
        # for p in self.model.model.parameters():
        #     print(p.shape, 'required_grad', p.requires_grad, 'grad != None', p.grad is not None, ', sum grad:', p.grad.abs().sum().item())
        # print('model_fine.parameters')
        # for p in self.model.model_fine.parameters():
        #     print(p.shape, 'required_grad', p.requires_grad, 'grad != None', p.grad is not None, ', sum grad:', p.grad.abs().sum().item())

        if self.grad_clip:
            clip_grad_value_(self.model.parameters(), self.grad_clip)
        if self.grad_norm:
            clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        # # TODO delete 
        # print('after step:')
        # for p in self.model.parameters():
        #     print(p.shape, ', Not None: ', p.grad is not None, ', sum grad:', p.grad.sum().item())

        if self.lr_scheduler is not None:
            self.lr_scheduler.step_batch(batch_counter)
        # # TODO delete
        # print('after parameters:')
        # after_para = list(p.clone() for p in self.model.parameters())
        # for p in after_para:
        #     print(p.shape, p.requires_grad, p.grad is not None, None if p.grad is None else p.grad.sum().item()) 
        # print('check parameters delta')
        # for p1, p2 in zip(before_para, after_para):
        #     print(p1.shape, p2.shape, (p1 - p2).sum().item())

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
        # # TODO delete
        # batch_counter = 0
        with torch.no_grad():
            logger.info('Begin Evaluation')
            callbacks.on_evaluate_begin()
            for batch in dataloader:
                callbacks.on_evaluate_batch_begin(batch)
                out = self.evaluate_batch(batch, predictor)
                callbacks.on_evaluate_batch_end(batch, out)
                # # TODO delete
                # batch_counter = batch_counter + 1
                # if batch_counter % 100 == 0:
                #     logger.info('Evaluate batch counter {}'.format(batch_counter))
            callbacks.on_evaluate_end()
            logger.info('End Evaluation')
        status = callbacks.get_evaluate_status()
        return status
        # return {}

    def evaluate(self, dataset, callbacks=None, predictor=None):
        self._build_trainer()
        if callbacks is None:
            callbacks = self.callbacks
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


@register_trainer('dist_trainer')
class DistributedTrainer(BasicTrainer):
    def __init__(self, args, model, predictor=None, train_dataset=None, valid_dataset=None, callbacks: list = None):
        super().__init__(args, model, predictor, train_dataset, valid_dataset, callbacks)

    def _prepare_building(self):
        devices = [d for d in self.args.devices.split(',') if d]
        if len(devices) > 0:
            assert len(devices) == 1
            self.device = 'cuda:{}'.format(devices[0])
            self.model.to(self.device)
        else:
            self.device = None
        logger.info('Device {}'.format(self.device))

    @classmethod
    def add_training_args(cls, parser, arglist=None):
        # use devices to specify gpus instead of CUDA_VISIBLE_DEVICES
        super().add_training_args(parser, arglist)
        parser.add_argument('--devices', type=str, default='', help='0,1')
        parser.add_argument('--dist-url', type=str, default='')
        parser.add_argument('--dist-backend', type=str, default='nccl')
