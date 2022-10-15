
import torch
from torch import nn

from . import register_model
from .base import PCTRFeatureMeta, PCTRModel


# @register_model('PCTRSENetSingleFeature')
class PCTRSENetSingleFeature(PCTRModel):
    def __init__(self, args, feature_meta: PCTRFeatureMeta, scale, activation='relu'):
        super().__init__(args, feature_meta)
        # self.embeddding_dim = embedding_dim
        self.scale = scale
        if activation == 'relu':
            ac = nn.ReLU()
        elif activation == 'sigmoid':
            ac = nn.Sigmoid()
        # Sparse feature only
        self.sparse_features = self.get_feature_list(type=['scalar', 'sparse_vector', 'var_len_vector'])
        n_sparse = len(self.sparse_features)
        print('n_sparse', n_sparse)
        print('sparse features', self.sparse_features)
        self.fc = nn.Sequential(
            nn.Linear(n_sparse, n_sparse // scale),
            nn.ReLU(),
            nn.Linear(n_sparse // scale, n_sparse),
            ac,
        )

    def forward(self, feature_emb_dict):
        values = []
        keys = []
        zeros = None
        # Append zeros to empty features.
        for k, _ in self.sparse_features:
            if k in feature_emb_dict:
                f_emb = feature_emb_dict[k]
                bsz = f_emb.size(0)
                f = f_emb.reshape(bsz, -1).mean(dim=1)  # avg pooling
                values.append(f)
                keys.append(k)
                if zeros is None:
                    zeros = f.new_zeros((bsz,))
            else:
                values.append(None)
                keys.append(None)
        for i, v in enumerate(values):
            if v is None:
                values[i] = zeros
        fin = torch.stack(values, dim=1)  # B x n_sparse
        weights = self.fc(fin)  # B x n_sparse
        # Multiply weights
        out = {}
        for i, (k, v) in enumerate(zip(keys, values)):
            if k is not None:
                v = feature_emb_dict[k]
                ori_shape = v.shape
                v = v.reshape(ori_shape[0], -1) * weights[:, i:i+1]  #  B x D
                out[k] = v.reshape(ori_shape)
        return out, weights
