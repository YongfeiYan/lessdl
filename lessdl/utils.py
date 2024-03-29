import torch
import glob
import importlib

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


def glob_with_comma(pattern):
    pattern = pattern.split(',')
    files = []
    for p in pattern:
        files.extend(glob.glob(p))
    return files


def acquire_attributes(obj, attrs, msg):
    """
    Check and get attributes
    """
    single = False
    if isinstance(attrs, str):
        single = True
        attrs = [attrs]
    r = []
    for a in attrs:
        if not hasattr(obj, a):
            raise RuntimeError('Attribute {} is not found, with message {}'.format(a, msg))
        r.append(getattr(obj, a))
    return r[0] if single else r


def acquire_keys(d, keys, msg):
    """Check and get required keys
    """
    single = False
    if isinstance(keys, str):
        single = True
        keys = [keys]
    r = []
    for k in keys:
        if k not in d:
            raise RuntimeError('Key {} is not found, with message {}'.format(k, msg))
        r.append(d[k])
    return r[0] if single else r


def import_class(class_pkg_name):
    if '.' not in class_pkg_name:
        return importlib.import_module(class_pkg_name)
    pkg = '.'.join(class_pkg_name.split('.')[:-1])
    pkg = importlib.import_module(pkg)
    name = class_pkg_name.split('.')[-1]
    return getattr(pkg, name)
