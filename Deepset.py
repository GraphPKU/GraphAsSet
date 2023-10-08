import torch
import torch.nn as nn
from utils import MLP
import MaskedReduce


class Set2Set(nn.Module):
    def __init__(self, hiddim: int, combine: str = "mul", aggr: str="sum", res: bool=True,  setdim: int=-2, **mlpargs) -> None:
        super().__init__()
        assert combine in  ["mul", "add"]
        self.mlp1 = MLP(hiddim, hiddim, hiddim, **mlpargs)
        self.mlp2 = MLP(hiddim, hiddim, hiddim, **mlpargs)
        self.setdim = setdim
        self.aggr = MaskedReduce.reduce_dict[aggr]
        self.res = res
        self.combine = combine

    def forward(self, x, mask):
        '''
        x (B, N, d)
        mask (B, N)
        '''
        x1 = self.mlp1(x)
        x1 = self.aggr(x1, mask.unsqueeze(-1), self.setdim).unsqueeze(self.setdim)
        x2 = self.mlp2(x)
        if self.combine == "mul":
            x1 = x1 * x2
        else:
            x1 = x1 + x2
        if self.res:
            x1 += x
        return x1

class Set2Vec(nn.Module):
    def __init__(self, hiddim: int, outdim: int, aggr: str="sum", setdim: int=-2, **mlpargs) -> None:
        super().__init__()
        self.mlp1 = MLP(hiddim, outdim, outdim, **mlpargs)
        self.mlp2 = MLP(outdim, outdim, outdim, **mlpargs)
        self.setdim = setdim
        self.aggr = MaskedReduce.reduce_dict[aggr]

    def forward(self, x, mask):
        '''
        x (B, N , d)
        mask (B, N)
        '''
        x1 = self.mlp1(x)
        x1 = self.aggr(x1, mask.unsqueeze(-1), self.setdim)
        return self.mlp2(x1)
    
