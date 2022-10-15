import os
import torch
import glob
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torch.utils.data.dataloader import default_collate
from functools import partial, update_wrapper
from itertools import zip_longest
from torchvision import datasets, transforms

from lessdl.data.vocab import Vocab
from lessdl.data import register_dataset


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """From fairseq.
    Convert a list of 1d tensors into a padded 2d tensor.
    pad_to_mutilple: pad的大小是这个参数的整数倍
    pad_to_length: pad到最小的长度.
    """
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


class BaseDataset:
    def collate(self, samples, pad_idx=0):
        """
        如果samples不是list, 则返回(可能是单个元素).
        samples是list, 每个元素是dict, 
        其中key为_开头的要用list进行batch.
        一维的LongTensor会被当作Tokens进行pad, pad_idx使用0.
        其余的使用pytorch默认的collate进行batch.
        """
        if not isinstance(samples, list):
            return samples
        out = {}
        elem = samples[0]
        for key in elem:
            if key.startswith('_'):
                out[key] = [x[key] for x in samples]
            elif isinstance(elem[key], torch.LongTensor) and elem[key].dim() == 1:
                out[key] = collate_tokens([x[key] for x in samples], pad_idx=pad_idx)
            else:
                out[key] = default_collate([x[key] for x in samples])
        return out
    
    @staticmethod
    def add_args(parser, arglist=None):
        pass

    @staticmethod
    def build(args, split=None):
        """
        split 决定加载train/valid/test数据集
        """
        raise NotImplementedError('')


class IterTextDataset(IterableDataset, BaseDataset):
    def __init__(self, name: str, file: str, vocab: Vocab, max_sent_size=None, 
        add_bos=True, add_eos=True):
        """
        name: 一个item的序列id的dict中的key.
        lazy的话, 只能访问iter, 否则的话, 能访问__item__
        """
        self.name = name
        self.file = file
        assert os.path.isfile(file), file
        self.vocab = vocab
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_sent_size = max_sent_size
        self.collate = partial(BaseDataset.collate, self, pad_idx=self.vocab.pad())
        update_wrapper(self.collate, BaseDataset.collate)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            worker_id = 0
            num_workers = 1
        else:  # in a worker process
            # split workload
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        with open(self.file) as f:
            for i, line in enumerate(f):
                if i % num_workers == worker_id:
                    yield self._prepro_line(line, i)

    def _prepro_line(self, line, ith_line):
        words = line.split()
        index = [self.vocab.word_to_index(w) for w in words]
        truncate = 0
        if self.max_sent_size and len(words) > self.max_sent_size:
            words, index = words[:self.max_sent_size], index[:self.max_sent_size]
            truncate = 1
        if self.add_bos:
            index.insert(0, self.vocab.bos())
        if self.add_eos:
            index.append(self.vocab.eos())
        return {
            self.name: torch.LongTensor(index),
            self.name + '_len': len(index), 
            '_' + self.name + '_raw': words,
            '_id': ith_line,
            '_size': len(index),
            self.name + '_truncate': truncate,
        }

    @staticmethod
    def add_args(parser, arglist=None):
        parser.add_argument('--vocab-path', type=str, required=True)
        parser.add_argument('--language', type=str, default='src')
        parser.add_argument('--file', type=str, required=True)
        parser.add_argument('--max-sent-size', type=int, default=512)
        parser.add_argument('--add-bos', action='store_true')
        parser.add_argument('--add-eos', action='store_true')
    
    @staticmethod
    def build(args, split=None):
        vocab = Vocab.from_file(args.vocab_path)
        return IterTextDataset(args.language, args.file, vocab, max_sent_size=args.max_sent_size, add_bos=args.add_bos, add_eos=args.add_eos)


class TextDataset(Dataset):
    def __init__(self, name: str, file: str, vocab: Vocab, max_sent_size=None, 
        add_bos=True, add_eos=True):
        """
            包装IterDataset成Dataset
        """
        self._dataset = IterTextDataset(name, file, vocab, max_sent_size, add_bos, add_eos)
        self.ds = [it for it in self._dataset]
        self.collate = self._dataset.collate
        self.vocab = self._dataset.vocab

    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):
        return len(self.ds)

    @staticmethod
    def add_args(parser, arglist=None):
        parser.add_argument('--vocab-path', type=str, required=True)
        parser.add_argument('--name', type=str, default='src')
        parser.add_argument('--file', type=str, required=True)
        parser.add_argument('--max-sent-size', type=int, default=512)
        parser.add_argument('--no-add-bos', action='store_true')
        parser.add_argument('--no-add-eos', action='store_true')
    
    @staticmethod
    def build(args, split=None):
        vocab = Vocab.from_file(args.vocab_path)
        return TextDataset(args.language, args.file, vocab, max_sent_size=args.max_sent_size, 
            add_bos=not args.no_add_bos, add_eos=not args.no_add_eos)


class PairIterDataset(IterableDataset, BaseDataset):
    def __init__(self, ds1: IterTextDataset, ds2: IterTextDataset):
        self.ds1 = ds1
        self.ds2 = ds2
        self._keys1 = None  # 记录是来自ds1的还是来自ds2的数据.
        self._keys2 = None

    def collate(self, samples):
        if not isinstance(samples, list):
            return samples
        b1, b2 = [], []
        for it in samples:
            b1.append({k: it[k] for k in self._keys1})
            b2.append({k: it[k] for k in self._keys2})
        b1 = self.ds1.collate(b1)
        b2 = self.ds2.collate(b2)
        b1.update(b2)
        return b1

    def __iter__(self):
        """
        zip用最长的, 两个数据集的长度不一致的话, 直接报错.
        """
        for it1, it2 in zip_longest(self.ds1, self.ds2):
            if self._keys1 is None:
                self._keys1 = set(it1.keys())
                self._keys2 = set(it2.keys())
            for key in it2:
                if key in self._keys1:
                    if key == '_size':
                        it1[key] = max(it1[key], it2[key])
                    else:
                        assert it1[key] == it2[key], (it1, it2, key, '不相同')
                else:
                    it1[key] = it2[key]
            yield it1

    @staticmethod
    def add_args(parser, arglist=None):
        parser.add_argument('--src-name', type=str, default='src')
        parser.add_argument('--src-file', type=str, required=True)
        parser.add_argument('--src-vocab-path', type=str, required=True)
        parser.add_argument('--src-max-sent-size', type=int, default=512)
        parser.add_argument('--tgt-name', type=str, default='tgt')
        parser.add_argument('--tgt-file', type=str, required=True)
        parser.add_argument('--tgt-vocab-path', type=str, required=True)
        parser.add_argument('--tgt-max-sent-size', type=int, default=512)
    
    @staticmethod
    def build(args, split=None):
        src_vocab = Vocab.from_file(args.src_vocab_path)
        src_data = IterTextDataset(args.src_name, args.src_file, src_vocab, max_sent_size=args.src_max_sent_size)
        tgt_vocab = Vocab.from_file(args.tgt_vocab_path)
        tgt_data = IterTextDataset(args.tgt_name, args.tgt_file, tgt_vocab, max_sent_size=args.tgt_max_sent_size)
        return PairIterDataset(src_data, tgt_data)


@register_dataset('translation_dataset')
class TranslationDataset(IterableDataset, BaseDataset):
    """
    add_bos add_eos只对src对文本起作用.
    """
    def __init__(self, data_dir, split, src_language, tgt_language, max_sent_size=512, add_bos=True, add_eos=True):
        self.src_file = os.path.join(data_dir, f'{split}.{src_language}')
        assert os.path.exists(self.src_file)
        self.tgt_file = os.path.join(data_dir, f'{split}.{tgt_language}')
        assert os.path.exists(self.tgt_file)
        self.src_vocab = Vocab.from_file(os.path.join(data_dir, 'vocab.' + src_language))
        self.tgt_vocab = Vocab.from_file( os.path.join(data_dir, 'vocab.' + tgt_language))
        self.padding_idx = self.src_vocab.pad()
        assert self.src_vocab.pad() == self.tgt_vocab.pad()
        self.max_sent_size = max_sent_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.src_ds = IterTextDataset('src', self.src_file, self.src_vocab, max_sent_size=max_sent_size, add_bos=True, add_eos=True)
        self.tgt_ds_in = IterTextDataset('tgt', self.tgt_file, self.tgt_vocab, max_sent_size=max_sent_size, add_bos=True, add_eos=False)
        self.tgt_ds_target = IterTextDataset('target', self.tgt_file, self.tgt_vocab, max_sent_size=max_sent_size, add_bos=False, add_eos=True)
        self.collate = partial(BaseDataset.collate, self, pad_idx=self.padding_idx)
        update_wrapper(self.collate, BaseDataset.collate)
    
    def __iter__(self):
        for src, tgt, target in zip_longest(self.src_ds, self.tgt_ds_in, self.tgt_ds_target):
            assert src is not None and tgt is not None and target is not None, 'src and tgt should be the same length.'
            ret = {}
            self._update_dict(ret, src)
            self._update_dict(ret, tgt)
            self._update_dict(ret, target)
            ret['_size'] = max(ret['src_len'], ret['tgt_len'])
            yield ret
    
    def _update_dict(self, ret, n):
        for k, v in n.items():
            if k not in ret:
                ret[k] = v
            else:
                if k == '_size':
                    ret[k] = max(ret[k], n[k])
                else:
                    assert ret[k] == v, f'values of {k} are not the same.'

    @staticmethod
    def add_args(parser, arglist=None):
        parser.add_argument('--data-dir', type=str)
        parser.add_argument('--src-language', type=str)
        parser.add_argument('--tgt-language', type=str)
        parser.add_argument('--max-sent-size', type=int, default=512)
        parser.add_argument('--no-add-bos', action='store_true')
        parser.add_argument('--no-add-eos', action='store_true')

    @staticmethod
    def build(args, split):
        assert args.src_language and args.tgt_language
        assert os.path.exists(args.data_dir), f'{args.data_dir} does not exist.'
        assert isinstance(split, str), split
        return TranslationDataset(args.data_dir, split, args.src_language, args.tgt_language, args.max_sent_size,
            not args.no_add_bos, not args.no_add_eos
        )

    def __repr__(self):
        return f'src_vocab: {self.src_vocab}\n' \
            f'tgt_vocab: {self.tgt_vocab}\n' \
            f'padding_idx: {self.padding_idx}\n' \
            f'add_bos: {self.add_bos}\nadd_eos: {self.add_eos}\nmax_sent_size: {self.max_sent_size}'


class FilesDataset(IterableDataset):
    def __init__(self, files):
        """
        files:
        """
        # TODO shuffle the files in multi-thread reading
        self.files = files
        # current reading index and file name in files
        self.cur_idx = None
        self.cur_file = None  

    def file_generator(self):
        """
        Return the generator of file conent
        """
        raise NotImplementedError()

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            worker_id = 0
            num_workers = 1
        else:  # in a worker process
            # split workload
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        for idx, file in list(enumerate(self.files))[worker_id::num_workers]:
            self.cur_idx = idx
            self.cur_file = file
            yield from self.file_generator()

    @staticmethod
    def add_args(parser, arglist=None):
        pass
    
    @staticmethod
    def build(args, split=None):
        raise NotImplementedError()


@register_dataset('imagenet_dataset')
class ImageNetDataset(datasets.ImageFolder):
    def __init__(self, dir, img_resize=256, img_crop=224) -> None:
        super().__init__()
        self.dir = dir
        self.img_resize = img_resize
        self.img_crop = img_crop
        super().__init__(dir,
            transforms.Compose([
                transforms.Resize(img_resize),
                transforms.CenterCrop(img_crop),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        )
    
    @staticmethod
    def add_args(parser, arglist=None):
        parser.add_argument('--data-dir', type=str, required=True)
        parser.add_argument('--img-resize', type=int, default=256)
        parser.add_argument('--img-crop', type=int, default=224)

    @staticmethod
    def build(args, split=None):
        data_dir = os.path.join(args.data_dir, split)
        return ImageNetDataset(data_dir, args.img_resize, args.img_crop)

