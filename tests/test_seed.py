
from simdltk import set_random_state
from tests.utils import build_trainer, get_testdata_trainer


set_random_state(13)
trainer = get_testdata_trainer('/tmp/test_seed1', rm_exp=True)
trainer.train()
print('evaluate')
trainer.evaluate(trainer.valid_dataset)
print('best metric', trainer.ckpt.best_metric)
m1 = trainer.ckpt.best_metric

set_random_state(13)
trainer = get_testdata_trainer('/tmp/test_seed2', rm_exp=True)
trainer.train()
print('evaluate')
trainer.evaluate(trainer.valid_dataset)
print('best metric', trainer.ckpt.best_metric)
m2 = trainer.ckpt.best_metric

assert abs(m1 - m2) < 0.00001, ('两次的ckpt metric差别太大', m1, m2)
print('OK!')
