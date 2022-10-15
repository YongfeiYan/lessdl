import unittest 
import tempfile 

from tests.utils import get_testdata_trainer
from lessdl.predictor.beam_search import BeamSearchPredictor


class TestBeamSearch(unittest.TestCase):
    def test_bs(self):
        d = tempfile.TemporaryDirectory().name
        trainer = get_testdata_trainer(d, rm_exp=True)
        trainer.args.epochs = 1
        trainer.train()

        device = trainer.device        
        args = trainer.args
        args.beam_size = 10
        args.max_steps = 2

        bs = BeamSearchPredictor.build(trainer.args, trainer.train_dataset, trainer.model)

        for batch in trainer.train_loader:
            topk = bs.predict(batch)
            print(topk['beam_topk'].shape)
            break

        print('bos', trainer.train_dataset.tgt_vocab.bos(), 'eos', trainer.train_dataset.tgt_vocab.eos())


if __name__ == '__main__':
    unittest.main()
