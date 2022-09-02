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


def glob_with_comma(pattern):
    pattern = pattern.split(',')
    files = []
    for p in pattern:
        files.extend(glob.glob(p))
    return files

