import os
import torch 
import argparse


def compare_model(m1, m2):
    m1 = torch.load(m1)
    m2 = torch.load(m2)
    # print('m1.keys', list(m1.keys()))
    # print('m2.keys', list(m2.keys()))
    for k in m1.keys():
        if k not in m2:
            print('k', k, 'in m1 but not found in m2')
        else:
            a1 = m1[k]
            a2 = m2[k]
            diff = (a1 - a2).abs().sum().item()
            if diff > 0.00001:
                print('k', k, 'diff', diff)
    for k in m2.keys():
        if k not in m1:
            print('k', k, 'in m2 but not found in m1')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('f1', type=str)
    parser.add_argument('f2', type=str)
    args = parser.parse_args()
    print('args:', args)

    compare_model('/tmp/m1.pt', '/tmp/m2.pt')

    pt1 = torch.load(args.f1)
    pt2 = torch.load(args.f2)
    out1 = pt1['outputs']
    out2 = pt2['outputs']
    print('pt1 keys', pt1['keys'])
    print('pt2 keys', pt2['keys'])
    print('pt1 index', pt1['outputs'][1])
    print('pt2 index', pt2['outputs'][1])
    print('pt1 - pt2 , x', (out1[0] - out2[0]).abs().sum())
    print('pt1 - pt2 , target', (out1[2] - out2[2]).abs().sum())
    print('pt1 - pt2 , logits', (out1[3] - out2[3]).abs().sum())
    print('pt1 - pt2 , logits diff', (out1[3] - out2[3]).abs().sum(dim=-1))
    print('pt1 - pt2 , loss', (out1[4] - out2[4]).abs().sum())
    print('pt1 - pt2 , loss diff', (out1[4] - out2[4].abs()))

    from tests.utils.tensor_op import compare_sorted_dict_tensors
    a = torch.load('local/image/debug2/grad_rank1.pt')
    b = torch.load('local/image/torch_example-debug/grad_rank1.pt')
    compare_sorted_dict_tensors(a, 'a', b, 'b')
    print('compare after grad', '\n' * 5)
    a = torch.load('local/image/debug2/grad1_rank1.pt')
    b = torch.load('local/image/torch_example-debug/grad1_rank1.pt')
    compare_sorted_dict_tensors(a, 'a', b, 'b')


"""
Usage
python -u examples/image_classification/comp_pt.py \
    local/image/debug2/data_train_epoch_0_batch_3_rank_1.pt \
    local/image/torch_example-debug/data_train_epoch_0_batch_3_rank_1.pt
"""


