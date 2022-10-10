
from tests.utils import get_testdata_trainer


trainer = get_testdata_trainer(rm_exp=True)
trainer.args.epochs = 1
trainer.train()

from simpledl.predictor.beam_search import BeamSearchPredictor

args = trainer.args
args.beam_size = 10
args.max_steps = 2

bs = BeamSearchPredictor.build(trainer.args, trainer.train_dataset, trainer.model)

for batch in trainer.train_loader:
    topk = bs.predict(batch)
    print(topk['beam_topk'].shape)
    break


print('bos', trainer.train_dataset.tgt_vocab.bos(), 'eos', trainer.train_dataset.tgt_vocab.eos())






print('OK!')
















