import torch
from torch import Tensor
from typing import Optional, List
import torch.nn as nn
from Norm import BatchNorm


def x2dims(x: Tensor):
    assert x.dim() == 2
    assert x.dtype == torch.int64
    ret = torch.max(x, dim=0)[0] + 1
    return ret.tolist()

def createemb(dim: int, emb_dim: int, zeropad: bool=False, max_norm: Optional[float]=None, orthoinit: bool=False):
    ret =  nn.Embedding(dim, emb_dim, max_norm=max_norm, padding_idx=0 if zeropad else None)
    if orthoinit:
        nn.init.orthogonal_(ret.weight.data)
    return ret

class MultiEmbedding(nn.Module):
    def __init__(self, emb_dim: int, dims: List[int], zeropad: bool = True, orthoinit=False, max_norm: Optional[float]=None, bn: bool=False, ln: bool=False, dropout: float=0.0):
        super().__init__()
        self.embedding_list = nn.ModuleList()

        for i, dim in enumerate(dims):
            self.embedding_list.append(createemb(dim, emb_dim, zeropad, max_norm, orthoinit))
        
        if ln:
            bn=False
            
        self.postemb = nn.Sequential()
        if ln:
            self.postemb.append(nn.LayerNorm(emb_dim, elementwise_affine=False))
        if bn:
            self.postemb.append(BatchNorm(emb_dim))
        if dropout > 0:
            self.postemb.append(nn.Dropout(dropout, inplace=True))


    def forward(self, x: Tensor):
        x_embedding = 0
        for i in range(x.shape[-1]):
            x_embedding += self.embedding_list[i](x.select(-1, i))
        return self.postemb(x_embedding)