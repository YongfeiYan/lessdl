import os
import shutil
import argparse
import torch
from itertools import chain
import copy

from lessdl.data.dataset import TranslationDataset
from lessdl import set_random_state


set_random_state(13)  # may fail to test if using another seed, such as 3

parser = argparse.ArgumentParser()
TranslationDataset.add_args(parser)
exp_dir = '/tmp/basictrainer'
# arg_line = f'python {__file__} --src-language en --tgt-language de --data-dir tests/data/mt-en-de'
arg_line = f'--exp-dir {exp_dir} --src-language en --tgt-language de --data-dir tests/data/mt-en-de --arch transformer_iwslt_de_en ' \
    f'--batch-size 3 --max-batch-tokens 0 --epochs 2  --max-samples-in-memory 1000000000'

arg_line = arg_line.split()
args, _ = parser.parse_known_args(arg_line)
# print(args)

train_data = TranslationDataset.build(args, 'train')
valid_data = TranslationDataset.build(args, 'valid')
test_data = TranslationDataset.build(args, 'test')
print('train_data', train_data)
n_lines = len(list(iter(train_data)))

from lessdl.model import get_model_cls, get_arch_arch
from lessdl.model.transformer import Transformer

parser.add_argument('--arch', type=str)
parser.add_argument('--model', type=str)
Transformer.add_args(parser)
args, _ = parser.parse_known_args(arg_line)
arch = get_arch_arch(args.arch)
arch(args)
print('args:\n', args)
model = Transformer.build(args, train_data)


accu_status = []
from lessdl.training.trainer import BasicTrainer
from lessdl.training.callbacks import Callback
class AccumulateCallback(Callback):
    """
    将训练和测试中出现的多个batch的记录累积成一个epoch的.
    """
    def __init__(self, train=False, evaluate=True):
        super().__init__(use_counter=False)
        self.train = train
        self.evaluate = evaluate
        self.train_status = {}
        self.evaluate_status = {}
    
    def add_to(self, data, s):
        for k, v in data.items():
            k = 'accu_' + k
            if k not in s:
                s[k] = [v]
            else:
                s[k].append(v)

    def concate(self, s):
        r = {}
        for k, v in s.items():
            # if isinstance(v[0], torch.Tensor):
            #     v = torch.cat(v, dim=0)  # tokens的size可能不一样
            # else:
            #     v = list(chain(*v))
            r[k] = v
        return r

    def on_train_batch_end(self, batch, logs=None):
        if not self.train:
            return
        self.add_to(batch, self.train_status)
        self.add_to(logs or {}, self.train_status)
    
    def reset_train_status(self):
        # TODO: delete
        accu_status.append(self.train_status)
        self.train_status = {}

    def get_train_status(self):
        if not self.train:
            return {}
        # r = self.concate(self.train_status)
        # return r
        return {}
    
    def on_evaluate_begin(self, logs=None):
        if not self.evaluate:
            return
        self.evaluate_status = {}

    def on_evaluate_batch_end(self, batch, logs=None):
        if not self.evaluate:
            return
        self.add_to(batch, self.evaluate_status)
        self.add_to(logs or {}, self.evaluate_status)

    def reset_evaluate_status(self):
        self.evaluate_status = {}

    def get_evaluate_status(self):
        if not self.evaluate:
            return {}
        r = self.concate(self.evaluate_status)
        return r


train_metrics = []
back_fn = None
back_log = None
def log_on_train_epoch_end(epoch, logs=None):
    print('log callback train metrics', back_log.train_metrics)
    print('log called', epoch, logs)
    train_metrics.append(copy.deepcopy(back_log._extract_metrics(back_log.train_metrics)))
    back_fn(epoch, logs)

print('parse argline:\n', arg_line)
BasicTrainer.add_args(parser, arg_line)
args, _ = parser.parse_known_args(arg_line)
print('trainer args:\n', args)
if os.path.exists(exp_dir):
    shutil.rmtree(exp_dir)
acb = AccumulateCallback(train=True, evaluate=True)
trainer = BasicTrainer(args, model, train_dataset=train_data, valid_dataset=valid_data, callbacks=[acb])

back_fn = trainer.log_cb.on_train_epoch_end
back_log = trainer.log_cb
trainer.log_cb.on_train_epoch_end = log_on_train_epoch_end

trainer.train()
trainer.evaluate(valid_data)


acb_train = acb.concate(accu_status[-1])
train_status = trainer.log_cb.get_train_status()
print(list(acb_train.keys()))
print(acb_train['accu_loss'])
print('train status', list(train_status.keys()))
print(train_status)
print('train metrics', train_metrics)

def get_acb_loss(acb_status):
    acbsumloss = 0
    acbsize = 0
    for i in range(len(acb_status['accu_sample_loss'])):
        bsz = len(acb_status['accu_src'][i])
        acbsumloss += acb_status['accu_sample_loss'][i] / acb_status['accu_sample_size'][i] * bsz
        acbsize += bsz
    acbloss = (acbsumloss / acbsize).item()
    return acbloss

acbloss = get_acb_loss(acb_train)
print('acbloss', acbloss, '\nnlines', n_lines, 
    '\nacb sample loss', acb_train['accu_sample_loss'], 
    '\nsum acb sample loss', sum(acb_train['accu_sample_loss'])
)
assert abs(acbloss - train_metrics[-1]['loss']) < 0.00001, (acbloss, train_metrics[-1]['loss'])


# 验证恢复训练后eval的结果一致
from lessdl.training.callbacks import default_format_status
acb_eval = acb.get_evaluate_status()
print('acb_eval', default_format_status(acb_eval), 'keys', list(acb_eval.keys()))
acbloss = get_acb_loss(acb_eval)
print('acb loss', acbloss)
trainer_eval_status = trainer.log_cb.get_evaluate_status()
print('eval status', default_format_status(trainer_eval_status))
assert abs(acbloss - trainer_eval_status['loss']) < 0.00001, (acbloss, trainer_eval_status['loss'])

arch(args)
old_optimizer = trainer.optimizer
old_model = model
model = Transformer.build(args, train_data)
trainer = BasicTrainer(args, model, train_dataset=train_data, valid_dataset=valid_data, callbacks=[acb])
last = trainer.ckpt.restore(last=True)
assert last == args.epochs - 1, (last, args.epochs)
trainer.evaluate(valid_data)
acb_eval2 = acb.get_evaluate_status()
acbloss2 = get_acb_loss(acb_eval2)
print('acb_eval2', default_format_status(acb_eval2), 'keys', list(acb_eval2.keys()))
assert abs(acbloss2 - trainer_eval_status['loss']) < 0.00001, (acbloss2, trainer_eval_status['loss'])


# 验证再训练一个epoch后, loss下降
# 保证重启训练后, 状态能恢复, loss能继续下降. -> 恢复后检查status是否变了, 然后
args.epochs = args.epochs + 1
trainer = BasicTrainer(args, old_model, train_dataset=train_data, valid_dataset=valid_data, callbacks=[acb])
trainer.train()
trainer.evaluate(train_data)
acb_eval3 = acb.get_evaluate_status()
acbloss3 = get_acb_loss(acb_eval3)
print('acb eval3', default_format_status(acb_eval3))
# print('old_optimizer', old_optimizer.state_dict())
# print('new_optimizer', trainer.optimizer.state_dict())
assert acbloss3 < acbloss2, (acbloss3, acbloss2)


print("OK!")
