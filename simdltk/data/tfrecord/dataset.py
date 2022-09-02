import os
import glob

from ..dataset import FilesDataset
from .reader import example_loader
from simdltk.utils import bool_flag, glob_with_comma


class TFRecordDataset(FilesDataset):
    def __init__(self, glob_pattern, use_mmap=False):
        # files = glob.glob(glob_pattern)
        files= glob_with_comma(glob_pattern)
        super().__init__(files)
        for f in files:
            assert os.path.exists(f), '{} does not exist.'.format(f)
        self.glob_pattern = glob_pattern
        self.use_mmap = use_mmap
        # print('files', files)

    def collate(self, samples, pad_idx=None):
        # print('TFRecordDataset samples', samples)
        if not isinstance(samples, list):
            return samples
        res = {}
        keys = set()
        for item in samples:
            for s in item.keys():
                keys.add(s)
        for k in keys:
            type_name = None
            values = []
            for s in samples:
                if k in s:
                    type_name = s[k][0]
                    values.append(s[k][1])
                else:
                    values.append(None)
            res[k] = (type_name, values)
        # print('TFRecordDataset res', res)
        return res

    def file_generator(self):
        for item in example_loader(self.cur_file, use_mmap=self.use_mmap):
            yield item

    @staticmethod
    def add_args(parser, arglist=None):
        raise NotImplementedError()

    @staticmethod
    def build(args, split=None):
        raise NotImplementedError()



