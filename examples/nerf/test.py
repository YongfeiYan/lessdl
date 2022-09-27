import torch
from simdltk.data.nerf_dataset import NeRFDataset
from simdltk.data.dataloader import DataLoader


def merge_index(dl):
    r = []
    for b in dl:
        r.append(b['index'])
    return torch.cat(r, 0)


def test_dataloader():
    """Run
    PYTHONPATH=. python examples/nerf/test.py --split train --data-dir local/nerf_synthetic/test_dataset --dataset-type blender --white-bkgd True
    """
    import argparse 
    # from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True)
    NeRFDataset.add_args(parser)
    args = parser.parse_args()
    print('Args', args)
    ds = NeRFDataset.build(args, args.split)
    dl = DataLoader(ds, batch_size=13, shuffle=True, drop_last=False, max_samples_in_memory=0)
    idx1 = merge_index(dl)
    idx2 = merge_index(dl)
    print('idx1.shape', idx1.shape)
    print('idx2.shape', idx2.shape)
    print('head idx1', idx1[:10])
    print('head idx2', idx2[:10])
    diff = (idx1 - idx2).abs().sum().item()
    print('diff', diff)
    assert sorted(idx2.int().tolist()) == list(range(len(idx2)))
    assert len(idx2) == len(ds), ('idx2 len:', len(idx2), 'ds len: ', len(ds))
    assert diff > 0, diff
    # for batch in dl:
    #     print('batch rays', batch['rays'].shape)
    #     print('batch target', batch['target'].shape)
    #     assert tuple(batch['rays'].shape[1:]) == (2, 3)
    #     assert tuple(batch['target'].shape[1:]) == (3,)
    #     break


def test_trainer():
    """DDPTraier

    """
    import argparse
    from simdltk.training.trainer_inc import DDPTrainer
    import torch 
    parser = argparse.ArgumentParser()
    DDPTrainer.add_args(parser)
    args = parser.parse_args()
    m = torch.nn.Linear(3, 4)
    trainer = DDPTrainer.build(args, m, 1, 1)
    trainer.train()


if __name__ == '__main__':

    test_trainer()
