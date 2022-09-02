import torch
import random
from collections import Counter

from simdltk.data.vocab import Vocab
from simdltk.data.dataset import TextDataset, IterTextDataset, BaseDataset, PairIterDataset
from simdltk.data.dataloader import DataLoader


### 先创建个词表
def build_vocab(file):
    with open(file) as f:
        lines = f.readlines()
        text = ' '.join(lines).split()
        n_lines = len(lines)
    cnt = Counter(text)
    v = Vocab(cnt, min_freq=2)
    return v

file = 'tests/data/en.txt'
v = build_vocab(file)
v.save_to_file('/tmp/t.vocab')
print(v)


### 测试基本的读取
ds = TextDataset('en', file, v)
n_lines = len(open(file).readlines())
assert len(ds) == n_lines, (len(ds), n_lines)
print('n_lines', n_lines)
ds2 = IterTextDataset('en', file, v)
# 检查doc
assert ds.collate.__doc__ == BaseDataset.collate.__doc__
print(ds.collate.__doc__)


### IterTextDataset要和TextDataset读取的一致.
for it, it2 in zip(ds.ds, iter(ds2)):
    # print(it)
    tokens = it['en']
    text = it['_en_raw']
    assert len(tokens) == len(text) + 2 == it['en_len']
    assert (tokens[0], tokens[-1]) == (v.bos(), v.eos())
    for t, c in zip(tokens[1:-1], text):
        assert t == v.word_to_index(c), (t, c)
    assert (it['en'] - it2['en']).sum().item() == 0
    assert it['_en_raw'] == it2['_en_raw']
    assert it['en_len'] == it2['en_len']


### 测试多worker和max_samples_in_memeory的读取. 用于bucket.
dl = DataLoader(ds2, batch_size=1, shuffle=True, max_samples_in_memory=1000)
dl2 = DataLoader(ds2, batch_size=1, shuffle=True, max_samples_in_memory=3, num_workers=4)
dl = sorted(dl, key=lambda x: x['_id'][0])
dl2 = sorted(dl, key=lambda x: x['_id'][0])

def assert_batch_eq(b, b2, skip_keys=None):
    skip_keys = skip_keys or []
    for key in b:
        if key in skip_keys: 
            continue
        # print(key, '\n', b[key], '\n', b2[key])
        if isinstance(b[key], torch.Tensor):
            assert (b[key] - b2[key]).abs().sum().item() == 0, (b, b2)
        else:
            assert b[key] == b2[key], (b, b2, f'key {key} is not equal')

for b, b2 in zip(dl, dl2):
    assert_batch_eq(b, b2)


### 测试shuffle要一致.
dl3 = DataLoader(ds2, batch_size=1, shuffle=False, max_samples_in_memory=100, sort_key='en_len', num_workers=1)
dl = sorted(dl, key=lambda x: x['en_len'][0])
for b, b3 in zip(dl, dl3):
    assert_batch_eq(b, b3)

### 测试PairIterTextDataset
filede = 'tests/data/de.txt'
vde = build_vocab(filede)
print(vde)
dsde = IterTextDataset('de', filede, vde, max_sent_size=4)
de_list = list(dsde)
assert len(de_list) == n_lines, len(de_list)
max_de_len = max(de_list, key=lambda x: len(x['de']))['de_len']
assert max_de_len == 6, max_de_len
ds4 = PairIterDataset(ds2, dsde)
dl4 = DataLoader(ds4, batch_size=4, shuffle=False, max_samples_in_memory=1000)
dlen = DataLoader(ds2, batch_size=4, shuffle=False, max_samples_in_memory=1000)
dlde = DataLoader(dsde, batch_size=4, shuffle=False, max_samples_in_memory=1000)
for bende, ben, bde in zip(dl4, dlen, dlde):
    assert_batch_eq(ben, bende, skip_keys=['_size'])
    assert_batch_eq(bde, bende, skip_keys=['_size'])
assert len(list(dlde)) == len(list(dl4))
dlde = DataLoader(dsde, batch_size=1, num_workers=3, max_samples_in_memory=1000)
assert len(list(dlde)) == n_lines, (len(list(dlde)), n_lines)
assert [x['_id'][0] for x in dlde] == list(range(n_lines))



### 测试build dataset
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
fileen = 'tests/data/en.txt'
filede = 'tests/data/de.txt'
vocaben = 'tests/data/vocab.en'
vocabde = 'tests/data/vocab.de'
ven = build_vocab(fileen)
ven.save_to_file(vocaben)
vde = build_vocab(filede)
vde.save_to_file(vocabde)
dsen = IterTextDataset('src', fileen, ven, max_sent_size=4)
dsde = IterTextDataset('tgt', filede, vde)
args.src_name = 'src'
args.src_vocab_path = vocaben
args.src_file = fileen
args.src_max_sent_size = 4
args.tgt_name = 'tgt'
args.tgt_file = filede
args.tgt_vocab_path = vocabde
args.tgt_max_sent_size = 250
dsende = PairIterDataset.build(args)
ende = PairIterDataset(dsen, dsde)
for a, b in zip(ende, dsende):
    assert_batch_eq(a, b)


### 测试translation dataset
from simdltk.data import get_dataset_cls
cls = get_dataset_cls('translation_dataset')
args.data_dir = 'tests/data/mt-en-de'
args.src_language = 'en'
args.tgt_language = 'de'
args.max_sent_size = 4
args.no_add_bos = False
args.no_add_eos = False
ds = cls.build(args, 'train')
dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2, max_samples_in_memory=1000)

ids = []
for batch in dl:
    for i, id in enumerate(batch['_id']):
        ids.append(id)
        itgt = list(batch['tgt'][i])
        itarget = list(batch['target'][i])
        ilen = batch['tgt_len'][i]
        assert batch['tgt_len'][i] == batch['target_len'][i]
        assert itgt[0] == ds.tgt_vocab.bos()
        for j, e in enumerate(itarget):
            if j < ilen - 1:
                assert e == itgt[j + 1]
            elif j == ilen - 1:
                assert e == ds.tgt_vocab.eos()
            else:
                assert e == ds.padding_idx
ids.sort()
assert ids == list(range(len(ids))), (ids, list(range(len(ids))))
assert len(ids) == n_lines


### 测试dataloader的max tokens
max_batch_tokens = 12
dl = DataLoader(ds, max_batch_tokens=max_batch_tokens, shuffle=True, max_samples_in_memory=12, sort_key='_size', num_workers=2)

ids = []
max_size = 0
for batch in dl:
    # print(batch['src'], batch)
    assert batch['src'].size(0) * batch['src'].size(1) <= max_batch_tokens, f'max tokens in batch is larger than {max_batch_tokens}, shape {batch["src"].shape}'
    assert batch['tgt'].size(0) * batch['tgt'].size(1) <= max_batch_tokens
    ids.extend(list(batch['_id']))
    for i in range(1, len(batch['_id'])):
        assert batch['_size'][i-1] <= batch['_size'][i], f'_size is not increasing order: {batch["_size"]}'
    # print(batch['_size'])
    max_size = max(max_size, sum(batch['_size']))

assert max_size == max_batch_tokens

ids.sort()
assert ids == list(range(len(ids))), (ids, list(range(len(ids))), 'dataloader 读取的id不是全部文件的id')
assert len(ids) == n_lines


### End
print('OK!')
