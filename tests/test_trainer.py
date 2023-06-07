
import os 
import torch 
import unittest 
import tempfile
import logging 
# from torch.utils.data import Dataset

from lessdl.training.trainer import DDPTrainer
from lessdl.data import register_dataset
from lessdl.model import register_model
from lessdl.model.base import BaseModel
from lessdl.data.dataset import BaseDataset

from tests.utils.trainer import build_trainer


@register_dataset('test_trainer_py_ds')
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, ds_size) -> None:
        super().__init__()
        self.ds_size = ds_size
        self.tensor = torch.arange(ds_size).float() / ds_size
        self.target = torch.LongTensor([0] * (ds_size // 2) + [1] * (ds_size - ds_size // 2))
    
    def __getitem__(self, index):
        return {
            'x': self.tensor[index],
            'target': self.target[index],
        }
    
    def __len__(self):
        return self.ds_size

    @staticmethod
    def add_args(parser, arglist=None):
        parser.add_argument('--ds-size', type=int, default=100)

    @staticmethod
    def build(args, split=None):
        return TestDataset(args.ds_size)


@register_model('test_trainer_py_model')
class TestModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.m = torch.nn.Linear(1, 2)
    
    def forward(self, x):
        x = x.reshape(-1, 1)
        logits = self.m(x)
        return {
            'logits': logits,
        }

    @staticmethod
    def add_args(parser, arglist=None):
        pass
    
    @classmethod
    def build(cls, args, dataset):
        return TestModel(args)


def get_argline(exp_dir, devices, batch_size):
    argline = """
    --exp-dir {} --devices {} --batch-size {} --epochs 2 --earlystopping 1
    --dataset test_trainer_py_ds 
    --model test_trainer_py_model
    --trainer ddp_trainer
    --dist-url tcp://127.0.0.1:8899
    --loss cross_entropy
    """.format(exp_dir, devices, batch_size)
    return ' '.join(argline.split())


class TestTrainer(unittest.TestCase):
    def test_ddp_trainer(self):
        exp_dir = tempfile.TemporaryDirectory().name 
        argline = get_argline(exp_dir, '0', 3)
        print('argline:', argline)
        trainer, *_ = build_trainer(argline)
        self.assertFalse(trainer.distributed, 'One device should not be distributed mode')
        self.assertIs(trainer._rank, None, 'non-ddp mode but got rank, trainer._rank {}'.format(trainer._rank))
        self.assertEqual(hasattr(trainer, 'optimizer'), True, 'non-distributed mode but no optimizer')
    
    def test_ddp_trainer_distributed(self):
        exp_dir = tempfile.TemporaryDirectory().name
        if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
            print('No enough GPUs, and skip test')
            return
        argline = get_argline(exp_dir, '0,1,2', 3)
        print('argline:', argline)
        trainer, *_ = build_trainer(argline)
        # check trainer.__init__
        self.assertTrue(trainer.distributed, '2 devices, distributed mode should be set')
        self.assertEqual(hasattr(trainer, 'optimizer'), False, 'distributed mode, optimizer should be built lazily')
        # check trainer.train 
        trainer.train()
        # logger = logging.getLogger()
        # log_file = tempfile.NamedTemporaryFile().name
        # hd = logging.FileHandler(log_file)
        # hd.setLevel(logging.INFO)
        # logger.addHandler(hd)
        # logger.info('Finished training, log file {}'.format(log_file))
        # print('after training, handlers', logger.handlers)
        # with open(log_file) as f:
        #     logs = '\n'.join(f.readlines())
        # self.assertIn('Rank 1 finished _train_worker and exit.', logs, 'Subprocess should exit aftering training')
        # self.assertIn('Rank 2 finished _train_worker and exit.', logs, 'Subprocess should exit aftering training')
        # self.assertNotIn('Rank 0 finished _train_worker and exit.', logs, 'Main process should not exit aftering training')
        # logger.removeHandler(hd)
        # TODO check data shuffle
        # import time 
        # time.sleep(10000)


if __name__ == '__main__':
    unittest.main()

