import torch
from torch import nn
import unittest 

from simpledl.data import get_dataset_cls
from simpledl.data.dataloader import DataLoader
from simpledl.loss.cross_entropy import CrossEntropy, LabelSmoothedCrossEntropy


class T:
    pass


class TestCE(unittest.TestCase):
    def test_ce(self):
        cls = get_dataset_cls('translation_dataset')
        args = T()
        args.cross_entropy_padding_idx = 0
        args.data_dir = 'tests/data/mt-en-de'
        args.src_language = 'en'
        args.tgt_language = 'de'
        args.max_sent_size = 4
        args.no_add_bos = False
        args.no_add_eos = False
        ds = cls.build(args, 'train')
        dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2, max_samples_in_memory=1000)

        task = T()
        task.padding_idx = ds.padding_idx
        args.sentence_avg = True

        ce = CrossEntropy.build(args, None, ds)
        args.label_smoothing = 0
        lce = LabelSmoothedCrossEntropy.build(args, None, ds)

        for batch in dl:
            t = torch.Tensor(batch['target'].size(0), batch['target'].size(1), len(ds.tgt_vocab)).normal_(std=0.01)
            out = {
                'logits': t
            }
            res = ce(batch, out)
            print(res)
            assert res['ntokens'] == batch['target_len'].sum().item()
            loss = nn.CrossEntropyLoss(ignore_index=task.padding_idx, reduction='sum')
            loss = loss(out['logits'].transpose(1, 2), batch['target'])
            assert abs(loss.item() - res['sample_loss'].item()) < 0.001, (loss.item(), res['sample_loss'].item())
            lloss = lce(batch, out)
            assert (lloss['loss'] - res['loss']).item() < 0.001, (lloss['loss'], res['loss'])


if __name__ == '__main__':
    unittest.main()
