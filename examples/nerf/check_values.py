
import argparse
import torch 


def check_values(apath, bpath, keys):
    import torch 
    a = torch.load(apath)
    b = torch.load(bpath)
    if not keys:
        keys = list(set(a.keys()) + set(b.keys()))
    for k, _ in keys:
        if k not in b:
            print(k, 'is not in', bpath)
        elif k not in a:
            print(k, 'is not in', apath)
        elif not isinstance(a[k], torch.Tensor):
            print(k, 'in', apath, 'is not torch.Tensor')
        elif not isinstance(b[k], torch.Tensor):
            print(k, 'in', bpath, 'is not torch.Tensor')
        else:
            if a[k].device != b[k].device:
                print('device is not the same')
                print('in', apath, ', device:', a[k].device)
                print('in', bpath, ', device:', b[k].device)
                a[k] = a[k].to(b[k].device)
            diff = (a[k] - b[k]).abs().item()
            if diff > 0.0001:
                print(k, 'is different, diff:', diff)
                print('in', apath, 'k:\n', a[k])
                print('in', bpath, 'k:\n', b[k])
            else:
                print(k, 'same!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('a', type=str)
    parser.add_argument('b', type=str)
    parser.add_argument('--keys', type=str)
    args = parser.parse_args()
    print(args)
    keys = [k for k in args.keys.split(',') if k]
    check_values(args.a, args.b, keys)
