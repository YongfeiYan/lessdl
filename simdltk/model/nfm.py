import torch
from torch.nn import functional as F
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pad_sequence

from torch.nn import Linear, Sequential, ReLU
from simdltk.model.base import BaseModel, PCTRModel
from simdltk.module.embedding import build_embedding
from simdltk.module.fm import nfm
from simdltk.model import register_model, register_model_architecture
from simdltk.module.functions import length_to_mask
from simdltk.model.senet import PCTRSENetSingleFeature


@register_model('NFM')
class NFM(BaseModel):
    def __init__(self, args, vocabs, emb_dim, dense_dim, hidden_dim1, hidden_dim2):
        super().__init__(args)
        # self.embeddings = embeddings
        self.emb_dim = emb_dim
        self.dense_dim = dense_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.vocabs = vocabs
        self.build_embeddings()
        self.fc1 = Sequential(
            Linear(emb_dim + dense_dim, hidden_dim1),
            ReLU(),
            Linear(hidden_dim1, hidden_dim1), 
            ReLU(),
            Linear(hidden_dim1, hidden_dim2),
            ReLU(),
            Linear(hidden_dim2, 1)
        )
    
    def build_embeddings(self):
        embs = {}
        for k, v in self.vocabs.items():
            v = build_embedding(v, self.emb_dim)
            embs[k] = v
            # self.register_parameter('_emb_' + k, v)  # only Parameter is valid, Module is not valid
            self.add_module('_emb_' + k, v)
        self.embs = embs

    def forward(self, dense_features, sparse_features):
        """
        [dense/sparse] features: {
            key: f,
            key.length:
            key.field: 
            # key.group: ...
            ...
        }
        f: a batch of tensor, eighter a list or a Tensor, whose first dimension is batch
        """
        values = []
        for k, v in sparse_features.items():
            if '.' in k:  # 去掉非feature向量
                continue
            assert k in self.vocabs, 'k {}'.format(k)
            values.append(self.embs[k](v))
        f_sparse = nfm(*values)  # B x Ds
        f_dense = []
        for k, v in dense_features.items():
            f_dense.append(v)
        fin = torch.cat(f_dense + [f_sparse], dim=1)  # B x D
        logits = self.fc1(fin).squeeze(1)  # B
        return {
            'logits': logits,
            'probs': torch.sigmoid(logits)
        }
    
    @staticmethod
    def add_args(parser, arglist=None):
        parser.add_argument('--emb-dim', type=int, default=192)
        parser.add_argument('--hidden-dim1', type=int, default=64)
        parser.add_argument('--hidden-dim2', type=int, default=64)
        parser.add_argument('--dense-dim', type=int, default=64)

    @classmethod
    def build(cls, args, dataset):
        vocabs = dataset.vocabs
        return NFM(args, vocabs, args.emb_dim, args.dense_dim, args.hidden_dim1, args.hidden_dim2)


@register_model('PCTRModelNFM')
class PCTRModelNFM(PCTRModel):
    def __init__(self, args, feature_meta, vocabs, emb_dim, dense_dim=None, hidden_dim1=None, hidden_dim2=None, seblock=None):
        super().__init__(args, feature_meta)
        # self.embeddings = embeddings
        self.sparse_features = self.get_feature_list(type=['scalar', 'sparse_vector', 'var_len_vector'])
        self.dense_features = self.get_feature_list(type=['embedding'])
        if dense_dim is None:
            dense_dim = sum([v.length for v, _ in self.dense_features])
        self.emb_dim = emb_dim
        self.dense_dim = dense_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.vocabs = vocabs
        self.build_embeddings()
        self.seblock = seblock
        self.fc1 = Sequential(
            Linear(emb_dim + dense_dim, hidden_dim1),
            ReLU(),
            Linear(hidden_dim1, hidden_dim1), 
            ReLU(),
            Linear(hidden_dim1, hidden_dim2),
            ReLU(),
        )
        self.out = Linear(hidden_dim2, 1)
    
    def build_embeddings(self):
        embs = {}
        for k, v in self.vocabs.items():
            v = build_embedding(v, self.emb_dim)
            embs[k] = v
            # self.register_parameter('_emb_' + k, v)  # only Parameter is valid, Module is not valid
            self.add_module('_emb_' + k, v)
        self.embs = embs

    def forward_embedding(self, batch):
        sparse_features = self.fetch_tensor(batch, self.sparse_features)
        dense_features = self.fetch_tensor(batch, self.dense_features)
        sparse_values = {}
        # length_info = {}
        for k, v in sparse_features.items():
            # if '.' in k:  # 去掉非feature向量
            #     continue
            assert k in self.vocabs, 'k {}'.format(k)
            # if isinstance(v, PackedSequence):
            #     v, _ = pad_packed_sequence(v, batch_first=True, padding_value=0)
            #     v = v[:, 1:]  # skip first dummy token
            # elif isinstance(v, (list, tuple)):
            #     v = pad_sequence(v, batch_first=True, padding_value=0)
            #     v = v[:, 1:]
            # elif v.is_sparse:
            #     # failed in multiprocessing DataLoader
            #     v = v.to_dense()
            # len_key = k + '.length'
            # len_key = '_' + k + '.length'
            # if len_key in batch:
            #     length = batch[len_key]
            #     # v = torch.tensor_split(v, length)
            #     v = torch.split(v, length)
            #     v = pad_sequence(v, batch_first=True, padding_value=0)
            v = self.embs[k](v)
            if k + '.length' in batch:
                length = batch[k + '.length']
                # length_info[k + '.length.valid'] = length.sum().detach()
                # length_info[k + '.length.total'] = length.max().detach() * len(length)
                mask = length_to_mask(length, v.size(1))  # B x N
                v = v * mask.unsqueeze(2)
            # values.append(v)
            sparse_values[k] = v  # sparse embeddigns
        f_dense = []  # list of dense embeddings
        for k, v in dense_features.items():
            f_dense.append(v)
        return f_dense, sparse_values  # dict of sparse embeddings, list of dense embeddigns
    
    def forward(self, batch):
        """
        [dense/sparse] features: {
            key: f,
            key.length:
            key.field: 
            # key.group: ...
            ...
        }
        f: a batch of tensor, eighter a list or a Tensor, whose first dimension is batch
        """
        logs = {}
        f_dense, sparse_values = self.forward_embedding(batch)
        for k, v in sparse_values.items():
            assert len(v.shape) in (2, 3), (k, v.shape)
        if self.seblock is not None:
            sparse_values, se_weights = self.seblock(sparse_values)
            logs['se_weights'] = se_weights
        for k, v in sparse_values.items():
            assert len(v.shape) in (2, 3), (k, v.shape)
        values = list(sparse_values.values())
        f_sparse = nfm(*values)  # B x Ds
        fin = torch.cat(f_dense + [f_sparse], dim=1)  # B x D
        # 
        fout = self.fc1(fin)
        logits = self.out(fout).squeeze(1)  # B
        probs = torch.sigmoid(logits)
        target = batch['target']
        loss = F.binary_cross_entropy(probs, target.float(), reduction='mean')
        logs.update({
            'logits': logits,
            'probs': probs,
            'loss': loss,
        })
        # logs.update(length_info)
        return logs
    
    @staticmethod
    def add_args(parser, arglist=None):
        parser.add_argument('--emb-dim', type=int, default=192)
        parser.add_argument('--hidden-dim1', type=int, default=64)
        parser.add_argument('--hidden-dim2', type=int, default=64)
        parser.add_argument('--dense-dim', type=int, default=64)
        parser.add_argument('--se-scale', type=int, default=0)
        parser.add_argument('--se-activation', type=str, default='relu')

    @classmethod
    def build(cls, args, dataset):
        vocabs = dataset.vocabs
        feature_meta = dataset.feature_meta
        if args.se_scale > 0:
            seblock = PCTRSENetSingleFeature(args, feature_meta, args.se_scale, args.se_activation)
        else:
            seblock = None
        return cls(args, feature_meta, vocabs, args.emb_dim, args.dense_dim, args.hidden_dim1, args.hidden_dim2, seblock=seblock)
