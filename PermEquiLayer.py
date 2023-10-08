from Deepset import Set2Set, Set2Vec
import torch
import torch.nn as nn
from typing import Callable
from torch_geometric.nn import Sequential as PygSequential
from utils import MLP


class PermEquiLayer(nn.Module):

    def __init__(self, hiddim: int, outdim: int, set2set: str, invout: bool,
                 numlayers: int, **kwargs) -> None:
        super().__init__()
        assert set2set in ["deepset", "transformer"]
        if set2set == "deepset":
            self.set2set = PygSequential(
                "x, mask",
                [(Set2Set(hiddim,
                          kwargs["combine"],
                          kwargs["aggr"],
                          res=kwargs["res"],
                          **kwargs["mlpargs1"]), "x, mask -> x")
                 for _ in range(numlayers)] + [(nn.Identity(), "x -> x")])
        elif set2set == "transformer":
            raise NotImplementedError
        if invout:
            self.set2vec = Set2Vec(hiddim,
                                   outdim,
                                   aggr=kwargs["pool"],
                                   **kwargs["mlpargs2"])
        else:
            self.set2vec = PygSequential(
                "x, mask", [(MLP(hiddim, outdim, outdim, **kwargs["mlpargs2"]), "x->x")])

    def forward(self, x, mask):
        '''
        x (B, N, d)
        mask (B, N)
        '''
        x = self.set2set(x, mask)
        x = self.set2vec(x, mask)
        return x