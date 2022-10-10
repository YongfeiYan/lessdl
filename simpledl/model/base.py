"""
主要定义一些类, 用于代表该类模型的构建方法.
"""
from torch import nn
from collections import namedtuple

from simpledl.data.dataset import PairIterDataset


class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    @staticmethod
    def add_args(parser, arglist=None):
        pass

    @classmethod
    def build(cls, args, dataset):
        raise NotImplementedError('')


class EncDecModel(BaseModel):
    def __init__(self, args, src_vocab, tgt_vocab):
        super().__init__(args)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    @classmethod
    def build(cls, args, dataset):
        assert hasattr(dataset, 'src_vocab')
        assert hasattr(dataset, 'tgt_vocab')
        return cls(args, dataset.src_vocab, dataset.tgt_vocab)


PCTRFeatureMeta = namedtuple('PCTRFeatureMeta', 
    ['type', 'dtype', 'emb_field', 'feature_group', 'length', 'max_length', 'comment'], 
    # defaults=[]
)
TYPE = ('scalar', 'sparse_vector', 'var_len_vector', 'embedding')
DTYPE = ('int64', 'float')


def check_types(type: list=None, dtype: list=None, emb_field: list=None, feature_group: list=None):
    if type:
        for t in type:
            assert t in TYPE, t
    if dtype:
        for t in dtype:
            assert t in DTYPE, t


class PCTRModel(BaseModel):
    def __init__(self, args, feature_meta):
        """
        feature_meta : {
            'f1': PCTRFeatureMeta
            'f2': {'type': 'xxx', 'dtype': '', ...}
        }
        type: scalar|sparse_vector|var_len_vector|embedding
        dtype: data type of feature, int64, float
        emb_field:
        feature_group:
        length: for vector and sparse_vector
        max_length: for var_len_vector
        """
        super().__init__(args)
        self.feature_meta = feature_meta
        self._feature_meta_list = sorted(list(feature_meta.items()))
        for k, v in feature_meta.items():
            assert isinstance(k, str) and isinstance(v, PCTRFeatureMeta), (k, v)
            assert v.type in TYPE
            assert isinstance(v.emb_field, int), v
            assert isinstance(v.feature_group, int), v

    def get_feature_list(self, type: list = None, dtype: list = None, emb_field: list = None, feature_group: list = None):
        """
        return [(k, PCTRFeatureMeta), ...]
        """
        check_types(type, dtype)
        res = []
        for k, v in self._feature_meta_list:
            if type and v.type not in type:
                continue
            if dtype and v.type not in dtype:
                continue
            if emb_field and v.emb_field not in emb_field:
                continue
            if feature_group and v.feature_group not in feature_group:
                continue
            res.append((k, v))
        return res
    
    def fetch_tensor(self, batch, feature_list):
        res = {}
        for k, _ in feature_list:
            if k in batch:
                res[k] = batch[k]
        return res

    def forward(self, batch):
        raise NotImplementedError()
