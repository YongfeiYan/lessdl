import random
import torch
from torch.utils.data.dataloader import _DatasetKind

import warnings

# class _IterDatasetWrapper(torch.utils.data.IterableDataset):
#     """
#     对IterableDataset进行包装, 每次不是按照顺序读取下一个元素, 而是先读取max_samples_in_memory, 
#     然后再看是否进行打乱, 或者按照sort_key进行排序, 用于bucketing.
#     """
#     def __init__(self, dataset, max_samples_in_memory, shuffle, sort_key):
#         assert not (shuffle and sort_key), 'shuffle and sort不能同时指定'
#         self.ds = dataset
#         self.max_samples_in_memory = max_samples_in_memory
#         self.shuffle = shuffle
#         self.sort_key = sort_key
#         self.collate = dataset.collate

#     def __iter__(self):
#         mem = []
#         for it in self.ds:
#             mem.append(it)
#             if len(mem) == self.max_samples_in_memory:
#                 yield from self._emit_mem(mem)
#                 mem = []
#         if mem:
#             yield from self._emit_mem(mem)

#     def _emit_mem(self, mem):
#         if self.sort_key:
#             mem = sorted(mem, key=self.sort_key)
#         elif self.shuffle:
#             random.shuffle(mem)
#         for it in mem:
#             yield it


class DataLoader(torch.utils.data.DataLoader):
    """
    TODO: 按照长度排序后, 仍然可以shuffle batch
    无论是否是iterable dataset, 都要支持tokens预取然后排序
    """
    def __init__(self, dataset, batch_size=None, max_batch_tokens=None, 
        shuffle=False, max_samples_in_memory=0, sort_key=None, 
        sampler=None, batch_sampler=None, num_workers=0,
        # collate_fn=None,
        pin_memory=False, 
        # drop_last=False, 
        timeout=0,
        worker_init_fn=None, multiprocessing_context=None):
        # if isinstance(dataset, torch.utils.data.Dataset):
        #     dataset = _IterDatasetWrapper(dataset, max_samples_in_memory, shuffle, sort_key)
        # 不再进行 shuffle, 不需要collate_fn
        # assert collate_fn is None, '只允许使用dataset.collate'
        assert not isinstance(dataset, torch.utils.data.IterableDataset) or max_samples_in_memory, 'IterableDataset requires max_samples_in_memory > 0'
        self.buffer_batch_size = 0
        batch_size = batch_size or None  # convert 0 to None
        assert batch_size or max_batch_tokens, f'batch size if specified via batch size or max_batch_tokens'
        if batch_size and max_batch_tokens:
            warnings.warn(f'batch_size {batch_size} and max_batch_tokens {max_batch_tokens} cannot be specified at the same time. Use max_batch_size only.')
        if sort_key and not shuffle:
            warnings.warn('指定sort key, 不指定shuffle, 数据有bias, 每次都是从小到大排序')
        if max_samples_in_memory:
            assert batch_size is None or batch_size <= max_samples_in_memory, f'batch size {batch_size} >= max samples in memory {max_samples_in_memory}'
            self.buffer_batch_size = batch_size
            batch_size = None
        if 0 < max_samples_in_memory < 100000:
            warnings.warn('max_samples_in_memory {} 如果太小的话，shuffle不均匀，可能导致性能下降，尽量都加载到内存中。'.format(max_samples_in_memory))
        self.max_batch_tokens = max_batch_tokens
        self.max_samples_in_memory = max_samples_in_memory
        self.sort_key = sort_key
        assert sort_key is None or isinstance(sort_key, str), f'sort_key is None or str, but got {sort_key}'
        self.shuffle = shuffle
        super().__init__(dataset, batch_size, shuffle=False, sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers, 
            collate_fn=dataset.collate,
            pin_memory=pin_memory, drop_last=False, timeout=timeout, worker_init_fn=worker_init_fn, 
            multiprocessing_context=multiprocessing_context
        )

    def __iter__(self):
        if self.max_samples_in_memory:
            mem = []
            for ele in super().__iter__():
                mem.append(ele)
                if len(mem) == self.max_samples_in_memory:
                    yield from self._batch_mem(mem)
                    mem = []
            if mem:
                yield from self._batch_mem(mem)
        else:
            yield from super().__iter__()

    def _sort_func(self, x):
        assert self.sort_key in x, f'sort_key {self.sort_key} is not in samples, whose keys are {list(x.keys())}'
        return x[self.sort_key]

    def _batch_mem(self, mem):
        if self.sort_key:
            mem = sorted(mem, key=self._sort_func)
        elif self.shuffle and self._dataset_kind == _DatasetKind.Iterable:  # non-iterable的dataset已经在dataloader父类super()中shuffle了
            random.shuffle(mem)
        batches = []
        cur_samples = []
        cur_max_tokens = 0

        def append_batch():
            nonlocal batches, cur_samples, cur_max_tokens
            batches.append(self.collate_fn(cur_samples))
            cur_samples = []
            cur_max_tokens = 0

        for ele in mem:
            if self.max_batch_tokens:
                assert '_size' in ele, f'_size should be specified in data item, but only found {list(ele.keys())}'
                if ele['_size'] > self.max_batch_tokens:
                    warnings.warn(f'Single data item size {ele["_size"]} is large than max_batch_tokens {self.max_batch_tokens}')
                cur_max_tokens = max(ele['_size'], cur_max_tokens)
                if (len(cur_samples) + 1) * cur_max_tokens > self.max_batch_tokens:
                    append_batch()
                    cur_max_tokens = ele['_size']
                cur_samples.append(ele)
            else:
                cur_samples.append(ele)
                if len(cur_samples) == self.buffer_batch_size:
                    append_batch()
        if cur_samples:
            append_batch()
        return self._may_shuffle_batches(batches)

    def _may_shuffle_batches(self, batches):
        if self.shuffle and \
            (self._dataset_kind == _DatasetKind.Iterable or self.sort_key):
            random.shuffle(batches)
        return batches


"""
PyTorch DataLoader 笔记

DataLoader
- Sampler: 给定dataset, 返回index的迭代, 用于索引数据(为什么不直接返回数据item的引用, 而是index, 在多进程收集数据的时候, 需要有index来进行排序, 不shuffle).
- Batch Sampler: 将sampler包装起来, 每次返回一个batch的index
对于iterable dataset, sampler和batch sampler如何工作的?
pin_memory mechanism?
mechanism of each argument?


需求: 
- 支持lazy loading, 办法: iterable dataset
- 支持多进程 √
- 支持bucket
    - 先在进程中取得多个element, 在主进程中排序和bucket
    - 在workers里面排序, bucket, 然后返回给主进程
"""





