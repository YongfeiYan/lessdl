import torch
import copy
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout, Linear, LayerNorm, ModuleList, Module


# class NFM(Module):
#     """
#     不同的特征 和的平方 - 平方的和, 一个特征可以是多个值, 每个值都参与到特征交叉的计算中.
#     """
#     def __init__(self):    
#         super().__init__()
    
#     def forward(self, features):
#         pass


def nfm(*features):
    """
    features: a list like [ f ... ]. 
        f is eather a Tensor with shape Batch x Dim for features with single value, or Batch x n x Dim for features with multiple values.
        变长序列的embedding, 要求用0 pad到同样的长度
    """
    s = 0
    sum_of_square = 0
    for f in features:
        if len(f.shape) == 2:
            f = f.unsqueeze(1)
        s = s + f.sum(dim=1)
        sum_of_square = sum_of_square + (f * f).sum(dim=1)
    return ((s * s) - sum_of_square ) / 2


class FwFM(Module):
    def __init__(self, feature_meta) -> None:
        super().__init__()

    def forward(self, *features):
        """
        Input: 
            sparse feature embeddings and its length, padded mebeddings shouble be 0s.
            non-batch computing
        """
        pass
    
    def init_parameters():
        # initialize fw weight 
        pass
