import unittest
import tempfile

from lessdl import set_random_state
from tests.utils import build_trainer, get_testdata_trainer


class TestSeed(unittest.TestCase):
    def test_seed(self):

        set_random_state(13)
        dir1 = tempfile.TemporaryDirectory().name
        print('dir1', dir1)
        trainer = get_testdata_trainer(dir1, rm_exp=True)
        trainer.train()
        print('evaluate')
        trainer.evaluate(trainer.valid_dataset)
        print('best metric', trainer.ckpt.best_metric)
        m1 = trainer.ckpt.best_metric

        set_random_state(13)
        dir2 = tempfile.TemporaryDirectory().name
        print('dir2', dir2)
        trainer = get_testdata_trainer(dir2, rm_exp=True)
        trainer.train()
        print('evaluate')
        trainer.evaluate(trainer.valid_dataset)
        print('best metric', trainer.ckpt.best_metric)
        m2 = trainer.ckpt.best_metric

        self.assertLess(abs(m1 - m2), 0.00001, ('The difference between two metrics is too large', m1, m2))


if __name__ == '__main__':
    unittest.main()
