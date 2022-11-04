import os
import sys
from os import path
import copy 
import math
import logging
from inspect import Parameter, signature
import torch
import torch.multiprocessing as mp 
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from functools import partial

from lessdl import bool_flag
from lessdl.data.dataloader import DataLoader
from lessdl.loss import get_loss_cls
from lessdl.training.utils import get_optimizer, get_lr_scheduler, save_args, move_to_device
from lessdl.training.callbacks import Checkpoint, CallbackList, LogCallback, LRStatCallback, Callback, BaseCallback
from lessdl.training import register_trainer, get_callback_cls


logger = logging.getLogger()


def forward_model(model, batch):
    if isinstance(model, DistributedDataParallel):
        # ddp forward basemodel
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


class BaseTrainer:
    """TODO: Add trainer abstract
    """
    def __init__(self, args, model, predictor=None, train_dataset=None, valid_dataset=None):
        """
        model/lr_scheduler/loss/callbacks ...
        """
        pass


@register_trainer('basictrainer')
class BasicTrainer:
    def __init__(self, args, model, predictor=None, train_dataset=None, valid_dataset=None, callbacks: list=None):
        # Setup attributes, 
        callbacks = callbacks or []
        assert isinstance(callbacks, list)
        self.callbacks = copy.copy(callbacks)  # For user
        self._trainer_callbacks = []  # For training procedure
        assert train_dataset is not None
        self.predictor = predictor
        self.args = args
        self.batch_size = args.batch_size
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self._ready = False
        # Try to build, _build_trainer can only depenend on the above attributes
        self._build_trainer(from_init=True)

    def _build_trainer(self, from_init=False):
        """Lazy building for distributed training
        from_init: whether build at the end of self.__init__
        Since train() may be called multiple times, _build_trainer support to build multiple times
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
        return self

    def _build_callbacks(self):
        # TODO: add log: max norm of gradients, grad clip and norm value, tokens/s, training time etc.
        # build user callbacks 
        for cb in self.args.callbacks.split(','):
            if not cb:
                continue
            cb_cls = get_callback_cls(cb)
            cb = cb_cls.build(self.args, self, self.callbacks, rank=None)
            self.callbacks.append(cb)
            logger.info('Add callback {}'.format(cb_cls.__name__))
        # build _trainer_callbacks
        args, callbacks, model = self.args, self._trainer_callbacks, self.model 
        lr_cb = LRStatCallback(optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)
        callbacks.append(lr_cb)
        log_cb = LogCallback(self.callbacks + callbacks, 
            log_every_n_batches=args.log_every_n_batches,
            log_every_n_epochs=args.log_every_n_epochs
        )
        # Put log callback and checkpoint callback in the end, as they depend on other callbacks status.
        callbacks.append(log_cb)
        self.log_cb = log_cb
        self.ckpt = Checkpoint(args.exp_dir, self.callbacks + callbacks, last_to_keep=args.last_to_keep, earlystopping=args.earlystopping)
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

    def _build_sampler(self, dataset, shuffle, drop_last, is_train=True):
        return None

    def _build_dataloader(self, dataset, is_train=True):
        """
        TODO: 分不同类型的数据, 构建不同loader class
        """
        shuffle = True if is_train and self.args.shuffle else False
        drop_last = True if is_train and shuffle else False
        sampler = self._build_sampler(dataset, shuffle, drop_last, is_train)
        if sampler is not None:
            shuffle = False
        if self.args.max_batch_tokens and self.batch_size:
            logger.warning(f'max_batch_tokens is specified, set batch_size=0')
            batch_size = 0
        else:
            batch_size = self.batch_size
        if not hasattr(dataset, 'collate'):
            # For torch dataset
            logger.info('Use torch.utils.data.DataLoader')
            dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.args.num_workers,
                sampler=sampler, drop_last=drop_last, pin_memory=self.args.pin_memory)
        else: 
            logger.info('Use customized DataLoader')
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
        last = self.ckpt.restore(last=True, map_location=self.device) or -1
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
            # # TODO delete 
            # print('set random state rank', self._rank)
            # from lessdl import set_random_state
            # set_random_state(self.args.seed)

            for batch in self.train_loader:            
                if cb_list.should_stop_training():
                    break
                batch = self.move_to_device(batch)
                cb_list.on_train_batch_begin({})
                out = self.train_batch(batch, batch_counter)
                cb_list.on_train_batch_end(batch, out)
                batch_counter += 1
                if self.args.eval_every_n_batches and batch_counter % self.args.eval_every_n_batches == 0:
                    self._evaluate_worker(self.valid_dataset, cb_list, self.predictor)
                    self.model.train()
            cb_list.on_train_epoch_end(e, {})
            if self.args.eval_every_n_epochs and (e + 1) % self.args.eval_every_n_epochs == 0:
                self._evaluate_worker(self.valid_dataset, cb_list, self.predictor)
            if self.lr_scheduler is not None:  # epoch step
                self.lr_scheduler.step()
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
        """Train a batch and add the output of model to the keys of batch.
        """
        # self.model.train()
        assert self.model.training == True
        self.optimizer.zero_grad()
        # batch = self.move_to_device(batch)
        out = self.forward_model(self.model, batch)
        loss_dict = self.loss(batch, out)
        out.update(loss_dict)
        if torch.isnan(out['loss']).any():
            logger.warn('NaN in loss, NaN count is {}'.format(torch.sum(torch.isnan(out['loss'])).item()))
            logger.warn('Skip batch because of NaN')
            return out
        out['loss'].backward()

        # # TODO delete 
        # from tests.utils.tensor_op import save_parameters
        # print('save optimizer')
        # grad_file = self.args.exp_dir + '/opt_rank{}.pt'.format(self._rank)
        # if not os.path.exists(grad_file):
        #     torch.save(self.optimizer.state_dict(), grad_file)
        #     logger.info('Optimizer rank {} {}'.format(self._rank, self.optimizer))

        # print('save grad')
        # grad_file = self.args.exp_dir + '/grad_rank{}.pt'.format(self._rank)
        # save_parameters(self.model, grad_file, with_grad=True, overwrite=False)
        # if not os.path.exists(grad_file):
        #     print('save grad', '-' * 20)
        #     grad = {}
        #     for k, v in self.model.named_parameters():
        #         grad[k + '.weight'] = v.data
        #         grad[k + '.grad'] = v.data.grad
        #     torch.save(grad, grad_file)

        if self.grad_clip:
            clip_grad_value_(self.model.parameters(), self.grad_clip)
        if self.grad_norm:
            clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step_batch(batch_counter)

        # # TODO delete
        # print('save grad')
        # grad_file = self.args.exp_dir + '/grad1_rank{}.pt'.format(self._rank)
        # save_parameters(self.model, grad_file, with_grad=True, overwrite=False)
        # if not os.path.exists(grad_file):
        #     print('save grad', '-' * 20)
        #     grad = {}
        #     for k, v in self.model.named_parameters():
        #         grad[k + '.weight'] = v.data
        #         grad[k + '.grad'] = v.data.grad
        #     torch.save(grad, grad_file)

        return out

    def evaluate_batch(self, batch, predictor=None):
        # TODO 使用predictor
        # self.model.eval()
        # with torch.no_grad():  # move to _evaluate
        assert self.model.training == False
        # batch = self.move_to_device(batch)
        if predictor is None:
            out = self.forward_model(self.model, batch)
            loss_dict = self.loss(batch, out)
            out.update(loss_dict)
        return out

    def _evaluate_worker(self, dataset, callbacks=None, predictor=None):
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
        callbacks.set_model(self.model)  # TODO also set rank
        dataloader = self._build_dataloader(dataset, is_train=False)
        self.model.eval()
        with torch.no_grad():
            logger.info('Begin Evaluation')
            callbacks.on_evaluate_begin()
            for batch in dataloader:
                batch = self.move_to_device(batch)
                callbacks.on_evaluate_batch_begin(batch)
                out = self.evaluate_batch(batch, predictor)
                callbacks.on_evaluate_batch_end(batch, out)
            callbacks.on_evaluate_end()
            logger.info('End Evaluation')
        status = callbacks.get_evaluate_status()
        return status

    def evaluate(self, dataset, callbacks=None, predictor=None):
        """Evaluate on dataset
        DONOT call this func in train() as evaluate() may have settings conflict with train(), such as spawn multiple processes.
        """
        self._build_trainer()
        if callbacks is None:
            callbacks = self.callbacks + self._trainer_callbacks
        else:
            callbacks = callbacks + [self.log_cb]
        predictor = predictor or self.predictor
        return self._evaluate_worker(dataset, callbacks, predictor)

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
            help='The format is the same as optimzier, such as: inverse_sqrt,warmup_updates=4000,warmup_end_lr=5e-4'
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
        parser.add_argument('--batch-size', type=int, metavar='N', default=0)
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
    def add_callbacks_args(cls, parser, arglist=None):
        parser.add_argument('--callbacks', type=str, default='', help='callback names such as acc_cb,speed_cb the order matters')
        args, _ = parser.parse_known_args(arglist)
        for cb in args.callbacks.split(','):
            if not cb:
                continue
            # add callback args
            cb_cls = get_callback_cls(cb)
            cb_cls.add_args(parser, arglist)

    @classmethod
    def add_args(cls, parser, arglist=None):
        cls.add_dataloader_args(parser, arglist)
        cls.add_loss_args(parser, arglist)
        cls.add_lr_scheduler_args(parser, arglist)
        cls.add_optimizer_args(parser, arglist)
        cls.add_training_args(parser, arglist)
        cls.add_callbacks_args(parser, arglist)

    @classmethod
    def build(cls, args, model, train_data, valid_data, callbacks: list = None):
        return cls(args, model, train_dataset=train_data, valid_dataset=valid_data, callbacks=callbacks)


class DataSamplerEpochSetter(BaseCallback):
    def __init__(self, args, trainer, precursors=None, rank=None, dist_sampler=None):
        self.dist_sampler = dist_sampler
        super().__init__(args, trainer, precursors, rank)

    def on_train_epoch_begin(self, epoch, logs=None):
        logger.info('DataSamplerEpochSetter rank {} set epoch {}, dist_sampler rank {}'.format(self.rank, epoch, self.dist_sampler.rank))
        self.dist_sampler.set_epoch(epoch)
        return super().on_train_epoch_begin(epoch, logs)


class DistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """Support set_default_tensor_type
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
        if args.devices is None:
            devices = [] if not torch.cuda.is_available() else [str(i) for i in range(torch.cuda.device_count())]
        else:
            devices = [x for x in args.devices.split(',') if x]
        logger.info('Devices {}'.format(devices))
        self.distributed = len(devices) > 1 or args.nnodes > 1
        self._nnodes = args.nnodes
        self._node_rank = args.node_rank
        self._local_rank = None
        self._rank = None
        self._world_size = len(devices) * args.nnodes  # total processes
        self._local_devices = devices
        self._dist_url = dist_url
        self._dist_backend = dist_backend
        self._info_prefix = ''
        self._train_data_sampler = None
        self.default_tensor_type = default_tensor_type
        super().__init__(args, model, predictor, train_dataset, valid_dataset, callbacks)
        # self.batch_size = self.batch_size * len(devices) if self.distributed else self.batch_size
    
    def _build_trainer(self, from_init=False):
        # Postpone to build in distributed mode
        if from_init and self.distributed:
            return self
        return super()._build_trainer(from_init=False)

    def _prepare_building(self):
        if self._rank is not None:
            self._info_prefix = 'Rank {} - '.format(self._rank)
            self.device = torch.device('cuda:{}'.format(self._local_devices[self._local_rank]))
        elif len(self._local_devices) > 0:
            assert len(self._local_devices) == 1, 'Only one device is supported in non-ddp mode, but got {}'.format(self._local_devices)
            self.device = 'cuda:{}'.format(self._local_devices[0])
        else:
            self.device = None
        if self.default_tensor_type:
            logger.info(self._info_prefix + 'set default tensor type to {}'.format(self.default_tensor_type))
            torch.set_default_tensor_type(self.default_tensor_type)
        if self.device is not None:
            torch.cuda.set_device(self.device)
            self.model.to(self.device)
            if self.distributed:
                gpu = int(self._local_devices[self._local_rank])
                logger.info(self._info_prefix + 'use gpu {}'.format(gpu))
                self.model = DistributedDataParallel(self.model, device_ids=[gpu])
        logger.info(self._info_prefix + 'Device {}'.format(self.device))
        a = torch.Tensor([0.0])
        logger.info(self._info_prefix + 'Default device: {}, default float: {}'.format(a.device, a.dtype))

    def _build_callbacks(self):
        # Add train data sampler
        if self.distributed:
            assert self._train_data_sampler is not None, 'Sampler should be set before callbacks'
            self._trainer_callbacks.append(DataSamplerEpochSetter(self.args, self, None, None, self._train_data_sampler))
        # NOTE: call in BasicTrainer the setup log_callback and checkpoint checkpoint, which depends on
        # callbacks and _trianer_callback
        super()._build_callbacks()  
        # Set rank callbacks
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

    def _train_worker(self, local_rank):
        self._local_rank = local_rank
        self._rank = local_rank + len(self._local_devices) * self._node_rank
        dist.init_process_group(self._dist_backend, init_method=self._dist_url, world_size=self._world_size, rank=self._rank)
        self._build_trainer()
        super().train()
        # TODO support multiple calls
        # TODO evaluate multiprocess supporting
        # Finish training, stop subprocesses
        dist.destroy_process_group()

    def _worker(self, local_rank, *args, is_train=True):
        self._local_rank = local_rank
        self._rank = local_rank + len(self._local_devices) * self._node_rank
        dist.init_process_group(self._dist_backend, init_method=self._dist_url, world_size=self._world_size, rank=self._rank)
        self._build_trainer()
        if is_train:
            super().train(*args)
        else:
            super().evaluate(*args)
        # TODO support multiple calls
        # TODO evaluate multiprocess supporting
        # Finish training, stop subprocesses, all workers should finished, e.g. calling all_reduce to synchronize.
        dist.destroy_process_group()

    def train(self):
        if not self.distributed:
            return super().train()
        assert self._dist_url, 'dist_url should be like: --dist-url tcp://127.0.0.1:PORT'
        logger.info('Launch DDP training, world_size {}, devices {}'.format(self._world_size, self._local_devices))
        if self.args.start_method == 'fork':
            mp.start_processes(self._train_worker, nprocs=len(self._local_devices), start_method=self.args.start_method)
        else:
            mp.spawn(self._train_worker, nprocs=len(self._local_devices))

    def evaluate(self, dataset, callbacks=None, predictor=None):
        if not self.distributed:
            return super().evaluate(dataset, callbacks, predictor)
        assert self._dist_url, 'dist_url should be like: --dist-url tcp://127.0.0.1:PORT'
        logger.info('Launch DDP evaluate, world_size {}, devices {}'.format(self._world_size, self._local_devices))
        mp.spawn(partial(self._worker, is_train=False), args=(dataset, callbacks, predictor), nprocs=len(self._local_devices))

    @classmethod
    def add_training_args(cls, parser, arglist=None):
        # use devices to specify gpus instead of CUDA_VISIBLE_DEVICES
        super().add_training_args(parser, arglist)
        parser.add_argument('--devices', type=str, default=None, help='e.g. 0,1 ; Use all available devices by default.')
        parser.add_argument('--dist-url', type=str, default='', help='e.g. tcp://127.0.0.1:PORT')
        parser.add_argument('--nnodes', type=int, default=1, help='Total number of nodes')
        parser.add_argument('--node-rank', type=int, default=0, help='ith node of all nodes')
        parser.add_argument('--dist-backend', type=str, default='nccl')
        parser.add_argument('--default-tensor-type', type=str, default='', help='e.g. torch.cuda.FloatTensor')
        parser.add_argument('--start-method', type=str, default='spawn', help='arg for mp.spawn')

    @classmethod
    def build(cls, args, model, train_data, valid_data, callbacks: list = None):
        return cls(args, model, train_dataset=train_data, valid_dataset=valid_data, callbacks=callbacks, dist_url=args.dist_url, dist_backend=args.dist_backend,  
            default_tensor_type=args.default_tensor_type
        )
