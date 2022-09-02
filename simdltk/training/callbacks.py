import os
import math
import logging
from copy import deepcopy, copy
from itertools import zip_longest
import torch
import json

from .lr_scheduler import Scheduler
from ..metrics.classification import binary_ctr_metrics, binary_auc

logger = logging.getLogger()


def default_format_status(status):
    items = sorted(list(status.items()))
    if not items:
        return ''
    s = []
    for k, v in items:
        sv = toscalar(v)
        if sv:
            s.append(f'{k}:{format_scalar(sv)}')
        elif isinstance(v, str):
            s.append(f'{k}:{v}')
    return ', '.join(s)


class Callback(object):
    """
    Callback用于训练的过程进行回调, 用于测试模型的训练的情况, 比如保存断点等等.
    每个Callback可以抽象成一个状态, 如果模型从中断的训练中恢复的话, 可以根据这个状态进行恢复.
    epoch/batch begin/end都是指的是training的时候.
    epoch|batch _ begin|end, 用于累积统计量
    get/reset/formart status 用于获得, 重置, 格式化当前的统计量, 
    打印一般只在LogCB中进行, Checkpoint中可以获取status, 用于比较最优的loss.
    CallbackList只是用于调用一系列的Callback
    """

    def __init__(self, use_counter=False):
        self.model = None
        self.use_counter = use_counter
        self.epoch_counter = 0
        self.batch_counter = 0

    def reset_train_status(self):
        pass

    def get_train_status(self):
        if not self.use_counter:
            return {}
        return {
            'epoch_counter': self.epoch_counter,
            'batch_counter': self.batch_counter
        }

    def set_train_status(self, status):
        if self.use_counter:
            self.epoch_counter = status.get('epoch_counter', 0)  # may be empty in descendants
            self.batch_counter = status.get('batch_counter', 0)
        return self

    def format_train_status(self):
        """
        将status进行表示成str, 用于输出等等.
        reset用于对多个batch累积训练信息的时候用.  
        随时获取训练过程的status. 
        """
        return default_format_status(self.get_train_status())

    def set_model(self, model):
        self.model = model
        return self

    def on_train_epoch_begin(self, epoch, logs=None):
        pass

    def on_train_epoch_end(self, epoch, logs=None):
        if self.use_counter:
            self.epoch_counter += 1

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        if self.use_counter:
            self.batch_counter += 1

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass
    
    def reset_evaluate_status(self):
        pass

    def get_evaluate_status(self):
        return {}

    def on_evaluate_begin(self):
        """
        开始前将状态初始化
        """
        self.reset_evaluate_status()

    def on_evaluate_batch_begin(self, batch, logs=None):
        pass

    def on_evaluate_batch_end(self, batch, logs=None):
        pass

    def on_evaluate_end(self, eval_logs=None):
        """
        因为evaluate的结果可能会被后续用到, 因此end的时候, 添加返回值
        """
        return self.get_evaluate_status()

    def should_stop_training(self):
        return False


class CallbackList(object):
    """Container abstracting a list of callbacks.

    # Arguments
        callbacks: List of `Callback` instances.
    # 
    """

    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = copy(callbacks)

    def append(self, callback):
        self.callbacks.append(callback)

    def reset_train_status(self):
        for c in self.callbacks:
            c.reset_train_status()

    def get_train_status(self):
        res = [c.get_train_status() for c in self.callbacks]
        res = [r for r in res if r]
        return res

    def set_train_status(self, status):
        for s, c in zip_longest(status, self.callbacks):
            c.set_train_status(s)
        return self

    def format_train_status(self):
        status = [c.format_train_status() for c in self.callbacks]
        return '\n'.join(status)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)
        return self

    def on_train_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.

        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_epoch_begin(epoch, logs)

    def on_train_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        """Called right before processing a batch.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a batch.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.

        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def __iter__(self):
        return iter(self.callbacks)

    def on_evaluate_begin(self):
        for c in self.callbacks:
            c.on_evaluate_begin()

    def on_evaluate_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for c in self.callbacks:
            c.on_evaluate_batch_begin(batch, logs)
    
    def on_evaluate_batch_end(self, batch, logs=None):
        logs = logs or {}
        for c in self.callbacks:
            c.on_evaluate_batch_end(batch, logs)

    def reset_evaluate_status(self):
        for c in self.callbacks:
            c.reset_evaluate_status()

    def get_evaluate_status(self):
        return [c.get_evaluate_status() for c in self.callbacks]

    def on_evaluate_end(self, eval_logs=None):
        eval_logs = eval_logs  or {}
        return [c.on_evaluate_end(eval_logs) for c in self.callbacks]

    def should_stop_training(self):
        return any(c.should_stop_training() for c in self.callbacks)


def toscalar(value):
    """
    将value转化成一个scalar, 包活 int float torch.Tensor
    否则返回None
    """
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, torch.Tensor) and len(value.shape) == 0:
        return value.item()
    elif isinstance(value, torch.Tensor):
        return None
    return None
    # raise RuntimeError('Unknow type of value {}'.format(value))


def format_scalar(value, ndigits=6):
    return f'{value:.{ndigits}f}'.rstrip('0').rstrip('.')


def update_dict(res, dict_list, overwride=True):
    assert isinstance(dict_list, list)
    dup_keys = set()
    for c in dict_list:
        for k, v in c.items():
            if k in res:
                dup_keys.add(k)
            if k not in res or overwride:
                res[k] = v 
    if dup_keys:
        logger.info(f'dup_keys: {sorted(list(dup_keys))}')


class LogCallback(Callback):
    """
    将多个callback的信息和训练的信息打印出来. 也把batch信息和epoch信息打印出来.
    """
    def __init__(self, callbacks, log_every_n_batches=500, log_every_n_epochs=1):
        super().__init__(use_counter=True)
        self.callbacks = copy(callbacks)
        self.log_every_n_batches = log_every_n_batches
        self.log_every_n_epochs = log_every_n_epochs
        self.train_metrics = {}
        self.eval_metrics = {}
        self.eval_batch_counter = 0

    def _add_to_metrics(self, batch, logs, metrics):
        """
        用batch_size加权计算
        """
        assert isinstance(logs, dict)
        batch_size = len(next(iter(batch.values())))
        for k, v in logs.items():
            v = toscalar(v)
            if v is not None:
                if k not in metrics:
                    metrics[k] = (batch_size, v * batch_size)
                else:
                    sb, sv = metrics[k]
                    metrics[k] = (sb + batch_size, sv + v * batch_size)
    
    def _extract_metrics(self, metrics):
        res = {}
        for k, v in metrics.items():
            res[k] = metrics[k][1] / metrics[k][0]
        return res

    def on_train_batch_end(self, batch, logs=None):
        """
        logs包含了模型的输出和loss的输出, 这里只记录能够转化成scalar的变量, 多个batch的结果进行平均.
        """
        super().on_train_batch_end(batch, logs)
        logs = logs or {}
        self._add_to_metrics(batch, logs, self.train_metrics)
        if self.log_every_n_batches and self.batch_counter % self.log_every_n_batches == 0:
            logger.info(self.format_train_status())
            self.reset_train_status()

    def on_train_epoch_end(self, epoch, logs=None):
        super().on_train_epoch_end(epoch, logs)
        if self.log_every_n_epochs and self.epoch_counter % self.log_every_n_epochs == 0:
            logger.info(self.format_train_status())
            self.reset_train_status()

    def format_train_status(self):
        metrics = self._extract_metrics(self.train_metrics)
        s = default_format_status(metrics)
        for c in self.callbacks:
            r = c.format_train_status()
            if not r:
                continue
            s = s + '\n' + r
        return super().format_train_status() + ' - ' + s

    def on_evaluate_batch_end(self, batch, logs=None):
        self.eval_batch_counter = self.eval_batch_counter + 1
        if self.log_every_n_batches and self.eval_batch_counter % self.log_every_n_batches == 0:
            logger.info('eval_batch_counter {}'.format(self.eval_batch_counter))
        self._add_to_metrics(batch, logs or {}, self.eval_metrics)

    def reset_evaluate_status(self):
        self.eval_metrics = {}
        for c in self.callbacks:
            c.reset_evaluate_status()

    def get_evaluate_status(self):
        return self._extract_metrics(self.eval_metrics)

    def on_train_end(self, logs=None):
        logger.info('Training end.')

    def on_evaluate_begin(self):
        self.eval_batch_counter = 0
        return super().on_evaluate_begin()

    def on_evaluate_end(self, logs=None):
        # 打印出评价的logs, 并且返回相应的值
        # add output logs
        status = self.get_evaluate_status()
        logs = logs or {}
        status.update(logs)
        message = f'Evaluation: {default_format_status(status)}'
        dup_keys = set()
        for c in self.callbacks:
            s = c.get_evaluate_status()
            if not s:
                continue
            for k, v in sorted(list(s.items())):
                if k in status:
                    dup_keys.add(k)
                status[k] = v
            message = message + '\n' + default_format_status(s)
        if dup_keys:
            logger.warn(f'Duplicate keys of callback evaluation status: {list(dup_keys)}')
        logger.info(message)

    def reset_train_status(self):
        self.train_metrics = {}
        for c in self.callbacks:
            c.reset_train_status()

    def get_train_status(self):
        status = super().get_train_status()
        status.update(self.train_metrics)
        return status

    def set_train_status(self, status):
        super().set_train_status(status)
        status.pop('batch_counter')
        status.pop('epoch_counter')
        self.metrics = deepcopy(status)


class Checkpoint(Callback):
    """Checkpoints and best evaluation checkpoint.
    TODO: 将optimizer和lr_scheduler的状态也保存.
    """
    def __init__(self, base_dir, callbacks, last_to_keep=1, earlystopping=10, cmp_key='loss', cmp_better='less'):
        """"""
        super().__init__(use_counter=True)
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.callbacks = copy(callbacks)
        self.last_to_keep = last_to_keep
        self.earlystopping = earlystopping
        self.ckpts_dir = os.path.join(base_dir, 'checkpoints')
        os.makedirs(self.ckpts_dir, exist_ok=True)
        self.stop_counter = 0
        self.cmp_key = cmp_key
        self.best_metric = None
        self.cmp_better = cmp_better
        self._checkpoint_prefix = []

    def _better_metric(self, new, old):
        if self.cmp_better == 'less':
            return new < old
        return new > old

    def should_stop_training(self):
        return self.stop_counter >= self.earlystopping

    def get_train_status(self):
        status = {
            'stop_counter': self.stop_counter,
            'best_metric': self.best_metric,
            'checkpoints': self._checkpoint_prefix,
            'callbacks': [c.get_train_status() for c in self.callbacks]
        }
        status.update(super().get_train_status())
        return status

    def set_train_status(self, status):
        super().set_train_status(status)
        self.stop_counter = status['stop_counter']
        self.best_metric = status['best_metric']
        self._checkpoint_prefix = status['checkpoints']
        for c, cs in zip_longest(self.callbacks, status['callbacks']):
            c.set_train_status(cs)

    def format_train_status(self):
        return ''

    def on_train_epoch_end(self, epoch, logs=None):
        super().on_train_epoch_end(epoch, logs)

    def on_evaluate_end(self, logs=None):
        # TODO: 使用子callbacks的status
        logs = logs or {}
        update_dict(logs, [c.get_evaluate_status() for c in self.callbacks])
        assert self.cmp_key in logs, f'{self.cmp_key} is not found in logs.keys(): {list(logs.keys())}.'
        metric = logs[self.cmp_key]
        metric = toscalar(metric)
        assert metric is not None, f'Cannot convert {logs[self.cmp_key]} to a scalar.'
        if self.best_metric is None or self._better_metric(metric, self.best_metric):
            self.best_metric = metric
            self.save_checkpoint(best=True)
            logger.info(f'Better performance, best {self.cmp_key}: {self.best_metric}')
            # 重置 stop counter
            self.stop_counter = 0
        else:
            logger.info(f'performance is not improved')
            self.stop_counter += 1
            self.save_checkpoint(best=False)
            # print logs
            logger.info(f'stop counter: {self.stop_counter}/{self.earlystopping}')

    def save_checkpoint(self, best=True):
        # save checkpoints and status, delete old checkpoints
        # 如果有旧的, 先进行删除
        if len(self._checkpoint_prefix) >= self.last_to_keep and self.last_to_keep > 0:
            prefix = self._checkpoint_prefix.pop(0)
            model_pt = prefix + '.pt'
            model_status = prefix + '.json'
            logger.info(f'remove {model_pt} and {model_status}')
            os.remove(model_pt)
            os.remove(model_status)

        def save_with_prefix(prefix):
            model_pt = prefix + '.pt'
            model_status = prefix + '.json'
            logger.info(f'saving checkpoints to {model_pt} and {model_status}')
            # print('TODO DELETE del state')
            state = self.model.state_dict()
            torch.save(state, model_pt)
            del state
            with open(model_status, 'w') as wt:
                json.dump(self.get_train_status(), wt)

        # 再把新的存储进去
        if self.last_to_keep > 0:
            prefix = os.path.join(self.ckpts_dir, f'checkpoint-{self.epoch_counter - 1}')
            self._checkpoint_prefix.append(prefix)
            save_with_prefix(prefix)
        # 如果保存best, 再存储一份, 不能存储链接, 旧的链接可能被删除.
        if best:
            prefix = os.path.join(self.base_dir, 'best')
            save_with_prefix(prefix)

    def load_checkpoint(self, prefix):
        model_pt = prefix + '.pt'
        model_status = prefix + '.json'
        logger.info(f'Load checkpoint from {model_pt}')
        state = torch.load(model_pt)
        self.model.load_state_dict(state)
        del state
        with open(model_status) as f:
            status = json.load(f)
            self.set_train_status(status)

    def parse_prefix_epoch_num(self, prefix):
        if '/' in prefix:
            prefix = prefix[prefix.rfind('/')+1:]  # 去除目录
        prefix = prefix.lstrip('checkpoint-')
        if prefix.isnumeric():
            return int(prefix)
        return None

    def find_last_checkpoint_prefix(self, save_dir):
        """
        return:
            prefix of the last checkpoint
        """
        last = None
        e = None
        for file in os.listdir(save_dir + '/checkpoints'):
            if file.startswith('checkpoint-') and file.endswith('.pt'):
                prefix = file.rstrip('.pt')
                c = self.parse_prefix_epoch_num(prefix)
                if c is not None and (e is None or e < c):
                    e = c
                    last = os.path.join(save_dir, 'checkpoints/' + prefix)
        return last

    def restore(self, best=False, last=False, save_dir=None):
        """
        TODO: 没有保存optimizer, lrscheduler等. 将status改成torch.save等形式, 而不是json等, 便于保存tensor.
        """
        logger.warn('checkpoint只保存了模型参数, 没有保存optimizer,lr scheduler等, 恢复的训练状态不对!!!')
        assert best or last, f'best or last should be specified.'
        if best:
            if save_dir:
                prefix = os.path.join(save_dir, 'best')
            else:
                prefix = os.path.join(self.base_dir, 'best')
        else:
            prefix = self.find_last_checkpoint_prefix(save_dir or self.base_dir)
        if not prefix or not os.path.exists(prefix + '.pt'):
            return None
        epoch = self.parse_prefix_epoch_num(prefix)
        self.load_checkpoint(prefix)
        return epoch


class LRStatCallback(Callback):
    """
    TODO: 只输出lr, 去除冗余的信息
    """
    def __init__(self, optimizer: torch.optim.Optimizer = None, lr_scheduler: Scheduler = None):
        super().__init__(use_counter=False)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def get_train_status(self):
        stat = {}
        if self.optimizer is not None:
            stat['optimizer'] = str(self.optimizer).replace('\n', '').replace('    ', ', ')
        if self.lr_scheduler is not None:
            stat['lr_scheduler'] = f'{type(self.lr_scheduler).__name__} lr {format_scalar(self.lr_scheduler.lr, ndigits=10)}'
        return stat

    def set_train_status(self, status=None):
        pass


class SpeedCallback(Callback):
    """
    TODO: n_tokens/s, n_samples/s, ...
    """
    def __init__(self):
        super().__init__(use_counter=False)
        self.train_status = {}
        self.eval_status = {}


class BinaryClassificationMetricsCallback(Callback):
    """
    auc, precision, recall etc. for binary classification tasks. 
    data_spec: may for multiple groups of metrics to evalute. such as prefix:labels_key:probs_key:filter,prefix2:labels_key2:probs_key2:filter,...
    """
    def __init__(self, auc=True, data_spec=None):
        super().__init__(True)
        self.auc = auc
        self.evaluate_status = {}
        self.data_spec = data_spec or ":target:probs:"
        self.group_train_labels = {}
        self.group_train_probs = {}
        self.group_train_filter = {}
        self.group_eval_labels = {}
        self.group_eval_probs = {}
        self.group_eval_filter = {}
        self.groups = []
        for gk in self.data_spec.split(','):
            keys = gk.split(':')
            assert len(keys) == 4, '{} should be like prefix:label_key:prob_key:filter'.format(keys)
            prefix, label_key, prob_key, filter = keys
            assert prefix not in [g[0] for g in self.groups], '{} is duplicated.'.format(prefix)
            self.groups.append((prefix, label_key, prob_key, filter))
        self._reset_labels_probs(self.group_train_labels, self.group_train_probs, self.group_train_filter)
        self._reset_labels_probs(self.group_eval_labels, self.group_eval_probs, self.group_eval_filter)

    def _get_metrics(self, labels, probs, filters):
        logger.info("Begin to calculate AUC")
        all_res = {}
        for prefix, label_key, prob_key, filter_key in self.groups:
            label_list = labels[prefix]
            prob_list = probs[prefix]
            if filter_key:
                filter_list = filters[prefix]
                label_list = [l for l, f in zip(label_list, filter_list) if f]
                prob_list = [l for l, f in zip(prob_list, filter_list) if f]
            res = binary_ctr_metrics(labels=label_list, predictions=prob_list)
            if self.auc:
                auc = binary_auc(labels=label_list, predictions=prob_list)
                res['auc'] = auc
            for k, v in res.items():
                all_res[prefix + k] = v
        logger.info("Calculate AUC end")
        return all_res

    def _update_labels_probs(self, labels, probs, filters, batch, out):
        res = {}
        res.update(batch)
        res.update(out)
        for prefix, label_key, prob_key, filter_key in self.groups:
            batch_labels = res[label_key].tolist()
            batch_probs = res[prob_key].tolist()
            if filter_key:
                batch_filters = res[filter_key].tolist()
                filters[prefix].extend(batch_filters)
            labels[prefix].extend(batch_labels)
            probs[prefix].extend(batch_probs)
    
    def _reset_labels_probs(self, labels, probs, filters):
        for prefix, label_key, prob_key, filter in self.groups:
            if prefix in labels:
                labels[prefix].clear()
                probs[prefix].clear()
                if filter:
                    filters[prefix].clear()
            else:
                labels[prefix] = []
                probs[prefix] = []
                if filter:
                    filters[prefix] = []

    def get_train_status(self):
        # return super().get_train_status()
        return {}

    def on_train_epoch_end(self, epoch, logs=None):
        super().on_train_epoch_end(epoch)
        status = self.get_train_status()
        status.update(self._get_metrics(self.group_train_labels, self.group_train_probs, self.group_train_filter))
        logger.info('BinaryClassificationMetrics - Train: ' + default_format_status(status))
        self._reset_labels_probs(self.group_train_labels, self.group_train_probs, self.group_train_filter)

    def on_train_batch_end(self, batch, out):
        super().on_train_batch_end(batch, out)
        self._update_labels_probs(self.group_train_labels, self.group_train_probs, self.group_train_filter, batch, out)
    
    def on_evaluate_end(self, eval_logs=None):
        super().on_evaluate_end(eval_logs)
        status = {}
        status.update(self._get_metrics(self.group_eval_labels, self.group_eval_probs, self.group_eval_filter))
        logger.info('BinaryClassificationMetrics - Evaluate: ' + default_format_status(status))
        self.evaluate_status = status
        self._reset_labels_probs(self.group_eval_labels, self.group_eval_probs, self.group_eval_filter)
    
    def on_evaluate_batch_end(self, batch, out):
        super().on_evaluate_batch_end(batch, out)
        self._update_labels_probs(self.group_eval_labels, self.group_eval_probs, self.group_eval_filter, batch, out)

    def get_evaluate_status(self):
        return self.evaluate_status


class AccumulatorCallback(Callback):
    """
    Accumulate outputs in each batch and save them to file.
    """
    def __init__(self, save_file_prefix=None, keys=''):
        super().__init__(use_counter=True)
        self.save_file_prefix = save_file_prefix
        self.keys = [k for k in keys.split(',') if k]
        logger.info('Keys to accumulate: {}'.format(self.keys))
        self.outputs = []

    def on_evaluate_begin(self):
        self.outputs.clear()
        self.outputs = [[] for _ in self.keys]
        return super().on_evaluate_begin()

    def on_evaluate_batch_end(self, batch, logs=None):
        # save each batch outputs
        logs = logs or {}
        out = {}
        out.update(batch)
        out.update(logs)
        for i, k in enumerate(self.keys):
            assert k in out, '{} not found in {}'.format(k, list(out.keys()))
            v = out[k]
            # only for torch.Tensor now
            if isinstance(v, torch.Tensor):  
                self.outputs[i].append(v.detach().cpu())
            else:
                raise NotImplementedError()
        return super().on_evaluate_batch_end(batch, logs)
    
    def _accumulated_outputs(self, reset=True):
        res = []
        for vs in self.outputs:
            print('len vs', len(vs), vs[0].shape, vs[0].dtype, vs[0].device)
            res.append(torch.cat(vs, dim=0))
        if reset:
            self.outputs.clear()
        return res

    def on_evaluate_end(self, eval_logs=None):
        super().on_evaluate_end(eval_logs)
        # accumulate and delete old list
        outputs = self._accumulated_outputs(reset=True)
        self.outputs.extend(outputs)
        # save to file
        if self.save_file_prefix:
            file = '{}_epoch_{}.pt'.format(self.save_file_prefix, self.epoch_counter)
            # os.makedirs(dir_of_file, exist_ok=True)
            logger.info('Saving outputs to {}'.format(file))
            torch.save({
                'keys': self.keys,
                'outputs': self.outputs,  # list of tensors
            }, file)
            logger.info('Saving finished')