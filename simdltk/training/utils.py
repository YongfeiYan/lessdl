import numpy as np
import torch
from torch import optim
import json

###################################################
# checkpoints
###################################################


# def save_checkpoint(filename, model, statistics, logger=None):
#     if logger:
#         logger.info('save checkpoint to ' + filename)
#     if filename and model:
#         statistics['state_dict'] = model.state_dict()
#         torch.save(statistics, filename)


# def load_checkpoint(filename, model=None, logger=None):
#     if logger:
#         logger.info('load checkpoint from ' + filename)
#     statistics = torch.load(filename)
#     if model:
#         model.load_state_dict(statistics['state_dict'])
#     return statistics


###################################################
# PyTorch utils 临时存放, 需要修改
###################################################


# def sparse_categorical_accuracy(logits, target, mean=True):
#     """logit: batch_size x C. target: batch_size with int64 dtype."""
#     arg = logits.argmax(dim=-1)
#     eq = torch.as_tensor(arg == target, dtype=torch.float32)
#     return eq.mean() if mean else eq


# def classification_accuracy(pred, gold_true):
#     """pred, gold_ture: torch tensors or numpy tensors."""
#     if isinstance(pred, torch.Tensor):
#         arg_max = torch.argmax(pred, dim=-1, keepdim=False)
#         return (arg_max == gold_true).float().mean().item()
#     return (np.argmax(pred, axis=-1) == gold_true).mean()


# import copy
# import inspect
# import logging
# import sys
# import warnings
# from contextlib import contextmanager
# from importlib import import_module
# from traceback import print_tb
# from typing import Any, Dict, List
# from collections import MutableMapping, OrderedDict
# import copy
# import json
# import logging
# import collections
# import os
# import warnings
# import argparse
# import time

# import numpy as np
# import yaml
# import six


# _LOGGER = None


# def get_logger(log_file, use_global=True):
#     """Set global _LOGGER if use_global."""
#     global _LOGGER

#     if use_global and _LOGGER:
#         return _LOGGER

#     logger = logging.getLogger(log_file)
#     logger.setLevel(logging.DEBUG)
#     fh = logging.FileHandler(log_file)
#     fh.setLevel(logging.DEBUG)
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     ch.setFormatter(formatter)
#     fh.setFormatter(formatter)
#     logger.addHandler(ch)
#     logger.addHandler(fh)

#     if use_global:
#         _LOGGER = logger
#     return logger


#####################################################################
# Metrics
#####################################################################


# class WeightedStatistic:
#     """将统计量进行加权，最终求平均。
#     值-权重 在累加前可以通过preprocessing进行处理，比如提取某些维，数据转换等等
#     平均后的值可以通过postprocessing进行处理，比如转化为熵
#     """
#     def __init__(self, name, init, postprocessing=None, preprocessing=None):
#         self.name = name
#         self.init = init
#         self.v = init
#         self.w = 0
#         self.preprocessing = preprocessing
#         self.postprocessing = postprocessing

#     def add(self, v, w):
#         if self.preprocessing is not None:
#             v, w = self.preprocessing(v, w)
#         self.v = self.v + v * w
#         self.w = self.w + w

#     def get(self):
#         # No accumulated value
#         if self.w == 0:
#             return 0
#         v = self.v / self.w
#         if self.postprocessing is not None:
#             v = self.postprocessing(v)
#         return v

#     def clear(self):
#         self.v = self.init
#         self.w = 0

#     def __repr__(self):
#         return '{} {:.5f}'.format(self.name, self.get())


# class BatchSizeWeightedStatistics:
#     """将output中对应key的值加入到相应的统计对象statistic中.
#     默认使用WeightedStatistic作为统计对象.
#     keys: list, tuple, dict，如果是list，tuple，则使用默认统计对象，否则使用dict中指定的统计对象
#     batch_size是通过每个batch中第一个数据获得的，不能通过输出获得因为输出可能是一个标量没有batch信息
#     """
#     def __init__(self, statistics):
#         if isinstance(statistics, (list, tuple)):
#             self.keys = set(statistics)
#             self.statistics = {k: WeightedStatistic(k, 0, None) for k in self.keys}
#         else:
#             self.keys = set(statistics.keys())
#             self.statistics = statistics
#         self._empty = True
#         self.clear()

#     @property
#     def empty(self):
#         return self._empty

#     def clear(self):
#         self._empty = True
#         # self.statistics = {k: WeightedStatistic(k, 0, None) for k in self.keys}
#         for stat in self.statistics.values():
#             stat.clear()

#     def add(self, data, outputs):
#         """
#         data: a dict of batch of examples.
#         outputs: a dict of keys and retrieved values.
#         """
#         self._empty = False
#         w = 1
#         keys = list(data.keys())
#         if len(keys) == 0:
#             warnings.warn('Empty data and set w = 1')
#         else:
#             w = len(data[keys[0]])
#         for k, v in self.statistics.items():
#             if k in outputs:
#                 v.add(outputs[k], w)

#     def pop(self):
#         """Return the dict and reset."""
#         d = self.get_dict()
#         self.clear()
#         return d

#     def get_dict(self):
#         statistics = self.statistics or {}
#         return {k: v.get() for k, v in statistics.items()}

#     def get(self, key):
#         return self.statistics[key].get()

#     def description(self, prefix='', digits=3):
#         format_str = '{} {:.' + str(digits) + 'f}'
#         return ', '.join([format_str.format(prefix + k, v) for k, v in self.get_dict().items()])


# class StatisticsList:
#     """统计数据列表，用statistic的name作为key存储成dict
#     """
#     def __init__(self, statistics=None):
#         statistics = statistics or []
#         if not isinstance(statistics, list):
#             statistics = [statistics]
#         self.statistics = {stat.name: stat for stat in statistics}
#         self._empty = True
#         self.clear()

#     @property
#     def empty(self):
#         return self._empty

#     def clear(self):
#         self._empty = True
#         # self.statistics = {k: WeightedStatistic(k, 0, None) for k in self.keys}
#         for stat in self.statistics.values():
#             stat.clear()

#     def add(self, data, outputs):
#         """
#         data: a dict of batch of examples.
#         outputs: a dict of keys and retrieved values.
#         """
#         self._empty = False
#         for stat in self.statistics.values():
#             stat.add(data, outputs)

#     def pop(self):
#         """Return the dict and reset."""
#         d = self.get_dict()
#         self.clear()
#         return d

#     def get_dict(self):
#         return {k: v.get() for k, v in self.statistics.items()}

#     def get(self, key):
#         return self.statistics[key].get()

#     def description(self, prefix='', digits=3):
#         format_str = '{} {:.' + str(digits) + 'f}'
#         return ', '.join([format_str.format(prefix + k, v) for k, v in self.get_dict().items()])


# class StatisticsKeyComparator(object):
#     def __init__(self, key='loss', cmp='less'):
#         if isinstance(cmp, str):
#             cmp = np.less if cmp == 'less' else np.greater
#         else:
#             assert callable(cmp), 'cmp should be less, greater or callable'
#         self.cmp = cmp
#         self.key = key

#     def __call__(self, new, old):
#         """Return true if new is better than old.
#         new, old: dict or statistics object which supports get method.
#         """
#         return self.cmp(new.get(self.key), old.get(self.key))


# def get_statistics(stat):
#     """
#     stat: [k1, k2], then use BatchSize
#     """
#     if isinstance(stat, str):
#         stat = [stat]
#     if isinstance(stat, (list, tuple)):
#         stat = BatchSizeWeightedStatistics(stat)
#     return stat


#####################################################################
# IO
#####################################################################


# def yaml_load(data_path):
#     with open(data_path) as f:
#         return yaml.load(f)


# def yaml_dump(data, data_path):
#     with open(data_path, 'w') as f:
#         yaml.dump(data, f)


# def jsonnet_load(data_path):
#     from _jsonnet import evaluate_file
#     return json.loads(evaluate_file(data_path))


# def jsonnet_dump(dictionary, file_path):
#     with open(file_path, 'w') as f:
#         json.dump(dictionary, f)


# def save_dict(d, filename):
#     """Save dict as yaml."""

#     def _map(v):
#         if type(v).__module__ == 'numpy':
#             return v.tolist()
#         else:
#             return v

#     yaml_dump({k: _map(v) for k, v in d.items()}, filename)


def save_args(args, file):
    with open(file, 'w') as wt:
        kwargs = args._get_kwargs()
        json.dump(kwargs, wt)


def load_args(args, file, overwrite=False):
    with open(file) as f:
        kwargs = json.load(f)
    for k, v in kwargs:
        if overwrite or getattr(args, k, None) is None:
            setattr(args, k, v)
    return kwargs


def move_to_device(batch, device):
    """
    将batch中非_开头key对应的value的Tensor都移动到device上
    """
    res = {}
    for k, v in batch.items():
        if k.startswith('_'):
            res[k] = v
        elif isinstance(v, torch.Tensor) or isinstance(v, torch.nn.utils.rnn.PackedSequence):
            res[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            res[k] = [vi.to(device) for vi in v]
        elif isinstance(v, tuple) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            res[k] = tuple(vi.to(device) for vi in v)
        else:
            res[k] = v
    # batch = {
    #     k: v.to(device) if not k.startswith('_') and (isinstance(v, torch.Tensor) or isinstance(v, torch.nn.utils.rnn.PackedSequence)) else v \
    #         for k, v in batch.items()
    # }
    # return batch
    return res


#######################################################
# optimizer的parse和构建, 来自facebook的toolkit
#######################################################
import inspect


def split_method_kwargs(s):
    """s: method,arg1=float1,arg2=float2..."""
    if "," in s:
        method = s[:s.find(',')]
        kwargs = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            # assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None, f'{split[0]},{split[1]}'
            kwargs[split[0]] = float(split[1])
    else:
        method = s
        kwargs = {}
    return method, kwargs


def get_optimizer(parameters, s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    # if "," in s:
    #     method = s[:s.find(',')]
    #     optim_params = {}
    #     for x in s[s.find(',') + 1:].split(','):
    #         split = x.split('=')
    #         assert len(split) == 2
    #         assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None, f'{split[0]},{split[1]}'
    #         optim_params[split[0]] = float(split[1])
    # else:
    #     method = s
    #     optim_params = {}
    method, optim_params = split_method_kwargs(s)

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
        if 'beta1' in optim_params:
            optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
            optim_params.pop('beta1', None)
            optim_params.pop('beta2', None)
    elif method == 'fairseq_adam':
        from fairseq.optim.adam import Adam
        optim_fn = Adam
        if 'beta1' in optim_params:
            optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
            optim_params.pop('beta1', None)
            optim_params.pop('beta2', None)
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    # elif method == 'adam_inverse_sqrt':
    #     optim_fn = AdamInverseSqrtWithWarmup
    #     optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.98))
    #     optim_params['warmup_updates'] = optim_params.get('warmup_updates', 4000)
    #     optim_params.pop('beta1', None)
    #     optim_params.pop('beta2', None)
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn(parameters, **optim_params)


def get_lr_scheduler(optimizer, s):
    """
    s:
        - none
        - inverse_sqrt,warmup_updates=4000,warmup_end_lr=5e-4
    """
    if s == 'none' or not s:
        return None
    method, kwargs = split_method_kwargs(s)
    from .lr_scheduler import InverseSquareRootSchedule
    if method == 'inverse_sqrt':
        return InverseSquareRootSchedule(optimizer, **kwargs)
    else:
        raise RuntimeError(f'{s} {kwargs}')












