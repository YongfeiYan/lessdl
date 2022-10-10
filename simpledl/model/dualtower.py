
import torch
from torch import nn
from torch.nn import functional as F

from .base import PCTRModel
from simpledl.model import register_model
from simpledl.module.embedding import build_embedding
from simpledl.module.fm import nfm


@register_model('DualTower')
class DualTower(PCTRModel):
    def __init__(self, args, vocabs, feature_meta, emb_dim, hidden_dim, out_dim):
        super().__init__(args, feature_meta)
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.vocabs = vocabs
        self.build_embeddings()
        # emb_field 0 
        self.sparse_features_0 = self.get_feature_list(type=['scalar', 'sparse_vector', 'var_len_vector'], emb_field=[0])
        self.dense_features_0 = self.get_feature_list(type=['embedding'], emb_field=[0])
        self.dense_dim_0 = sum([v.length for _, v in self.dense_features_0])
        self.tower_0 = nn.Sequential(
            nn.Linear(self.dense_dim_0 + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        # emb_field 1
        self.sparse_features_1 = self.get_feature_list(type=['scalar', 'sparse_vector', 'var_len_vector'], emb_field=[1])
        self.dense_features_1 = self.get_feature_list(type=['embedding'], emb_field=[1])
        self.dense_dim_1 = sum([v.length for _, v in self.dense_features_1])
        self.tower_1 = nn.Sequential(
            nn.Linear(self.dense_dim_1 + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
 
    def build_embeddings(self):
        embs = {}
        for k, v in self.vocabs.items():
            v = build_embedding(v, self.emb_dim)
            embs[k] = v
            # self.register_parameter('_emb_' + k, v)  # only Parameter is valid, Module is not valid
            self.add_module('_emb_' + k, v)
        self.embs = embs
    
    def forward_tower(self, batch, sparse_feature_list, dense_feature_list, tower):
        sparse_features = self.fetch_tensor(batch, sparse_feature_list)
        dense_features = [batch[k] for k, _ in dense_feature_list if k in batch]
        values = []
        for k, v in sparse_features.items():
            assert k in self.vocabs, 'k {}'.format(k)
            values.append(self.embs[k](v))
        f_sparse = nfm(*values)  # B x Ds
        if dense_features:
            f_sparse = torch.cat([f_sparse] + dense_features, dim=1)
        out = tower(f_sparse)
        return out

    def forward(self, batch):
        out0 = self.forward_tower(batch, self.sparse_features_0, self.dense_features_0, self.tower_0)
        out1 = self.forward_tower(batch, self.sparse_features_1, self.dense_features_1, self.tower_1)
        target = batch['target']
        probs = (torch.cosine_similarity(out0, out1) + 1) / 2  # map to 0~1, Dim: B
        loss = F.binary_cross_entropy(probs, target.float(), reduction='mean')
        return {
            'emb_0': out0,
            'emb_1': out1,
            'probs': probs,
            'loss': loss
        }

    @staticmethod
    def add_args(parser, arglist=None):
        parser.add_argument('--emb-dim', type=int, default=96)
        parser.add_argument('--hidden-dim', type=int, default=128)
        parser.add_argument('--out-dim', type=int, default=32)
    
    @classmethod
    def build(cls, args, dataset):
        return cls.build(args, dataset.vocabs, dataset.feature_meta, args.emb_dim, args.hidden_dim, args.out_dim)
