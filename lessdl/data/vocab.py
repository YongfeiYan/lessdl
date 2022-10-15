"""
实现简单的功能, 用于vocab到id到转换.
一个model可能需要多个不同的vocab, 比如character级别/word级别/不同语言的.
"""
from collections import defaultdict, Counter
from itertools import chain


class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    根据torchtext中的Vocab进行删除某些函数和功能得到
    增加了unk_idx设置，词表的词的个数，如果用len(stoi)可能会得到变化的值
    要求word是string!!
    """

    def __init__(self, counter: Counter, max_size=None, min_freq=1, specials=None,
        pad: str='<pad>', unk: str='<unk>', bos: str='<bos>', eos: str='<eos>'):
        """Create a Vocab object from a collections.Counter.
        """
        self._pad, self._unk, self._bos, self._eos = str(pad), str(unk), str(bos), str(eos)
        self.itos = [w for w in [pad, unk, bos, eos] if w and w != 'None']
        self.stoi = {v: k for k, v in enumerate(self.itos)}

        self.freq = counter.copy()
        counter = counter.copy()  # 不修改原参数

        specials = specials or []
        self._add_words(specials)

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self._add_words([word])

    def __eq__(self, other):
        for k in chain(self.itos, other.itos):
            if self.get_word_freq(k) != other.get_word_freq(k) or self.word_to_index(k) != other.word_to_index(k):
                return False
        if self.stoi != other.stoi or \
            self.itos != other.itos or \
            (self._pad, self._unk, self._bos, self._eos) != (other._pad, other._unk, other._bos, other._eos):
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def pad(self):
        return self.stoi[self._pad]
    
    def unk(self):
        return self.stoi[self._unk]
    
    def bos(self):
        return self.stoi[self._bos]
    
    def eos(self):
        return self.stoi[self._eos]
    
    def _add_words(self, words):
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

    # def extend(self, v):
    #     """
    #     合并两个词表
    #     v是Vocab实例
    #     """
    #     words = sorted(v.itos) if sort else v.itos
    #     self._add_words(words)
    #     for k, f in v.freq.items():
    #         if k not in self.freq:
    #             self.freq[k] = f

    def get_word_freq(self, w):
        return self.freq.get(w, 0)

    def word_to_index(self, w):
        return self.stoi.get(w, self.unk())

    def index_to_word(self, idx):
        return self.itos[idx]

    def save_to_file(self, file):
        with open(file, 'w') as wt:
            print(len(self.itos), self._pad, self._unk, self._bos, self._eos, file=wt)
            for w in self.itos:
                print(w, self.freq.get(w, 0), file=wt)

    @staticmethod
    def from_file(file, pad='<pad>', unk='<unk>', bos='<bos>', eos='<eos>'):
        cnt = Counter()
        specials = []
        with open(file) as f:
            line = f.readline()
            _, pad, unk, bos, eos = line.split()
            for line in f:
                line = line.split()
                if not line:  # 结尾空行
                    continue
                c = int(line[-1]) if len(line) > 1 else 0
                cnt[line[0]] = c
                specials.append(line[0])
        return Vocab(cnt, specials=specials, max_size=None, min_freq=0, pad=pad, unk=unk, bos=bos, eos=eos)

    def __repr__(self):
        return f'Vocab size {len(self)}, PAD {self._pad}, UNK {self._unk}, BOS {self._bos}, EOS {self._eos}'

