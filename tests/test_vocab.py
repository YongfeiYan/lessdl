from lessdl.data.vocab import Vocab
from collections import Counter

import unittest 
import tempfile 


class TestVocab(unittest.TestCase):
    def test_vocab(self):
        file = tempfile.NamedTemporaryFile().name
        print('vocab file:', file)
        
        # test vocab size 
        cnt = Counter({'a': 5, 'b': 4, 'c': 3, 'd': 2})
        v = Vocab(cnt)
        assert len(v) == 8
        v = Vocab(cnt, min_freq=3)
        assert len(v) == 7
        v = Vocab(cnt, specials=['d', '<pad>'], min_freq=3)
        assert len(v) == 8

        # test itos stoi
        v.save_to_file(file)
        v2 = Vocab.from_file(file)
        print(v.freq == v2.freq)
        print('v v2', v.freq, v2.freq)
        print(v.stoi == v2.stoi)
        print(v.stoi, v2.stoi)
        print(v.itos == v2.itos)
        print((v._pad, v._unk, v._bos, v._eos) == (v2._pad, v2._unk, v2._bos, v2._eos))
        assert v == v2

        for i, w in enumerate(v.itos):
            assert v.word_to_index(w) == i
            assert v.index_to_word(i) == w
            assert v.word_to_index('xxxx') == v.unk()
        assert (v.pad(), v.unk(), v.bos(), v.eos()) == (0, 1, 2, 3)

        # test save and load
        v = Vocab(cnt, bos=None, eos=None)
        v.save_to_file(file)
        v2 = Vocab.from_file(file)
        assert v == v2
        assert (v.pad(), v.unk(), v.word_to_index('<bos>') == v.unk(), v.word_to_index('<eos>') == v.unk())

        f = """6 <pad> <unk> None None
        a
        b
        c
        d
        """
        with open(file, 'wt') as wt:
            print(f, file=wt)
        v = Vocab.from_file(file)
        print(v)
        assert len(v) == 6
        assert v.unk() == 1
        assert v.word_to_index('d') == 5
        assert v.get_word_freq('d') == 0


if __name__ == '__main__':
    unittest.main()
