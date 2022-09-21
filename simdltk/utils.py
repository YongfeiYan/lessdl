import torch
import glob

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise RuntimeError("invalid value for a boolean flag")


def assert_no_nan(tensor):
    """
    Check Tensor or list of Tensor, skip those are not Tensor
    """
    if isinstance(tensor, (list, tuple)):
        for i, t in enumerate(tensor):
            if not isinstance(t, torch.Tensor):
                print('skip assert_no_nan with {}th element of tensor, type {}'.format(i, type(t)))
                continue
            assert t.isnan().sum().item() == 0, 'Nan found, {} element of tensor {}'.format(i, tensor)
    elif isinstance(tensor, torch.Tensor):
        assert tensor.isnan().sum().item() == 0, 'Nan found in {}'.format(tensor)
    else:
        print('skip assert_no_nan with type', type(tensor))


def get_required_keys(d, keys, msg):
    """
    TODO: for implicit dict interface with keys required
    """
    pass


def glob_with_comma(pattern):
    pattern = pattern.split(',')
    files = []
    for p in pattern:
        files.extend(glob.glob(p))
    return files
