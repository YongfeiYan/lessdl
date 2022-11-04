import os
import torch 
import argparse


def save_parameters(model, file, with_grad=True, overwrite=True):
    if not overwrite and os.path.exists(file):
        return 
    d = {}
    for k, v in model.named_parameters():
        d[k] = v
        if with_grad:
            d[k + '.grad'] = v.grad
    torch.save(d, file)


def compare_tensors(a, aname, b, bname):
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        print(aname, type(a), 'and', bname, type(b), 'should be torch.Tensor')
        return
    if a.device != b.device:
        print('device is not the same')
        print('in', aname, ', device:', a.device)
        print('in', bname, ', device:', b.device)
        a = a.to(b.device)
    diff = (a - b).abs().sum().item()
    if diff > 0.0001:
        print(aname, bname, 'is different, diff:', diff)
        # print('in', aname, 'k:\n', a)
        # print('in', bname, 'k:\n', b)
    else:
        print(aname, bname, 'same!!')


def compare_dict_tensors(a, aname, b, bname, keys=None):
    """Compare tensors in a and b.
    If keys are specified, only compare those keys.
    aname, bname are strings of a/b to print.
    """
    if not keys:
        keys = list(set(a.keys()) + set(b.keys()))
    for k, _ in keys:
        if k not in b:
            print(k, 'is not in', bname)
        elif k not in a:
            print(k, 'is not in', aname)
        elif not isinstance(a[k], torch.Tensor):
            print(k, 'in', aname, 'is not torch.Tensor')
        elif not isinstance(b[k], torch.Tensor):
            print(k, 'in', bname, 'is not torch.Tensor')
        else:
            compare_tensors(a[k], aname + ' key ' + k, b[k], bname + ' key ' + k)


def compare_sorted_dict_tensors(a, aname, b, bname):
    """Compare values of a, b by sorting their keys.
    """
    if len(a) != len(b):
        print('len', aname, len(a), '!= len', bname, len(b))
        return 
    akv = sorted(list(a.items()))
    bkv = sorted(list(b.items()))
    for (ak, av), (bk, bv) in zip(akv, bkv):
        compare_tensors(av, aname + ' key ' + ak, bv, bname + ' key ' + bk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('apath', type=str, )
    parser.add_argument('bpath', type=str, )
    parser.add_argument('--keys', type=str, defalut='')
    args = parser.parse_args()
    keys = [k for k in args.keys.split(',') if k]
    a, b = torch.load(args.apath), torch.load(args.bpath)
    compare_dict_tensors(a, args.apath, b, args.bpath, keys)

"""
Usage
python tests/utils/comp_torch_tensor.py a.pt b.pt x,logits
"""

