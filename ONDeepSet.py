import torch
from typing import Final
import torch.nn as nn
from MaskedReduce import reduce_dict
from InputEncoder import QInputEncoder
from PermEquiLayer import PermEquiLayer
from utils import MLP
import torch.nn.functional as F
from PiOModel import svMix, VMean, VNorm, Imod


class GlobalAggr(nn.Module):
    def __init__(self, hiddim, **kwargs) -> None:
        super().__init__()
        self.scalar = PermEquiLayer(hiddim, hiddim, "deepset", True, **kwargs["permlayer"])
        self.linv1 = nn.Linear(hiddim, hiddim, False)
        self.linv2 = nn.Linear(hiddim, hiddim, False)
        self.linv3 = nn.Linear(hiddim, hiddim, False)

    def forward(self, s, v, nodemask):
        '''
        s (B, N, d)
        v (B, N, M, d)
        nodemask (B, N)
        gsize (B, )
        return (B, d), (B, M, d), (B, M, M, d)
        '''
        gs = self.scalar.forward(s, nodemask)
        gv = self.linv3(v.sum(dim=1))
        gv2 = torch.einsum("bnad,bncd->bacd", self.linv1(v), self.linv2(v))
        return gs, gv, gv2

class Tprod(nn.Module):
    def __init__(self, hiddim, **kwargs) -> None:
        super().__init__()
        self.lin1 = nn.Linear(hiddim, hiddim)
        self.lin2 = nn.Linear(hiddim, hiddim)
        self.lin3 = nn.Linear(hiddim, hiddim)
        self.lins = MLP(hiddim, hiddim, hiddim, **kwargs["mlp"])
        self.linv1 = nn.Linear(hiddim, hiddim, False)
        self.linv2 = nn.Linear(hiddim, hiddim, False)
        self.linv3 = nn.Linear(hiddim, hiddim, False)

    def forward(self, s, v, gs, gv, gv2):
        '''
        s (B, N, d)
        v (B, N, M, d)
        nodemask (B, N)
        return (B, d), (B, M, d), (B, M, M, d)
        '''
        v_vv2 = self.linv1(torch.einsum("bnmd,bmcd->bncd", v, gv2))
        v_vs = self.linv3(gs.unsqueeze(-2).unsqueeze(-2) * v)
        v_sv = self.linv2(s.unsqueeze(2)*gv.unsqueeze(1))
        v = (v_vv2 + v_sv + v_vs) / 3
        s_v2t = torch.einsum("bmmd->bd", gv2)
        s_vv = torch.einsum("bmd,bnmd->bnd", gv, v)
        s = (self.lin1(s_v2t)*gs).unsqueeze(-2)*self.lin2(s)*self.lin3(s_vv)
        s = self.lins(s)
        return s, v


class SimpleTprod(nn.Module):
    def __init__(self, hiddim, **kwargs) -> None:
        super().__init__()
        self.lin1 = MLP(hiddim, hiddim, hiddim, **kwargs["mlp"])
        self.lin2 = MLP(hiddim, hiddim, hiddim, **kwargs["mlp"])
        self.linv1 = nn.Linear(hiddim, hiddim, False)
        self.linv2 = nn.Linear(hiddim, hiddim, False)

    def forward(self, s, v, gs, gv, gv2):
        '''
        s (B, N, d)
        v (B, N, M, d)
        nodemask (B, N)
        return (B, d), (B, M, d), (B, M, M, d)
        '''
        v_vv2 = self.linv1(torch.einsum("bnmd,bmcd->bncd", v, gv2))
        v = (v_vv2 + self.linv2(gv).unsqueeze(1)) / 1.414
        s = (self.lin1(s) + self.lin2(gs.unsqueeze(1))) / 1.414
        return s, v

class ONDeepSet(nn.Module):
    elres: Final[bool]
    num_layers: Final[int]
    num_tasks: Final[int]
    nodetask: Final[bool]
    gsizenorm: Final[float]
    def __init__(self,
                 featdim: int,
                 caldim: int,
                 hiddim: int,
                 outdim: int,
                 num_layers: int,
                 pool: str,
                 **kwargs) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_tasks = outdim
        self.elres = kwargs["elres"]
        self.nodetask = (pool=="none")
        self.pool = reduce_dict[pool]
        usesvmix = kwargs["usesvmix"]

        self.inputencoder = QInputEncoder(featdim, hiddim,
                                          **kwargs["inputencoder"])
        self.LambdaEncoder = PermEquiLayer(hiddim, hiddim, "deepset",
                                           False, **kwargs["l_model"])

        self.gaggrs = nn.ModuleList(
            [GlobalAggr(hiddim, **kwargs["gaggr"]) for _ in range(num_layers)]
        )
        
        self.svmixs = nn.ModuleList(
            [svMix(hiddim, **kwargs["svmix"]) if usesvmix else Imod() for _ in range(num_layers)]
        )
        if kwargs["simtprod"]:
            self.tprods = nn.ModuleList(
                [SimpleTprod(hiddim, **kwargs["tprod"]) for _ in range(num_layers)]
            )
        else:
            self.tprods = nn.ModuleList(
                [Tprod(hiddim, **kwargs["tprod"]) for _ in range(num_layers)]
            )

        self.predlin = MLP(hiddim, hiddim, outdim, **kwargs["predlin"])
        self.predln = nn.LayerNorm(outdim, elementwise_affine=False) if kwargs["outln"] else nn.Identity()

        self.vln = nn.Sequential(VMean(hiddim) if kwargs["vmean"] else nn.Identity(), VNorm(hiddim) if kwargs["vnorm"] else nn.Identity())
        self.elvln = nn.Sequential(VMean(hiddim) if kwargs["elvmean"] else nn.Identity(), VNorm(hiddim) if kwargs["elvnorm"] else nn.Identity())
        self.sln = nn.LayerNorm(hiddim, elementwise_affine=False) if kwargs["snorm"] else nn.Identity()
        self.gsizenorm = kwargs["gsizenorm"]

    def eigenforward(self, LambdaEmb, LambdaMask, U, X, nodemask):
        '''
        LambdaEmb (#graph, M, d1)
        LambdaMask (#graph, M)
        U (#graph, N, M)
        X (#graph, N, dx)
        nodemask (#graph, N)
        A (#graph, N, N, A)
        '''
        B, N, M = U.shape[0], U.shape[1], U.shape[2]
        gsize = N - torch.sum(nodemask.float(), dim=1)
        gsizenorm = torch.rsqrt_(gsize).pow_(self.gsizenorm).reshape(-1, 1, 1, 1)
        gsizenorm_v = gsizenorm.reshape(-1, 1, 1)
        LambdaEmb = self.LambdaEncoder(LambdaEmb, LambdaMask)  # LambdaEmb (#graph, M, d1)
        LambdaEmb = torch.where(LambdaMask.unsqueeze(-1), 0, LambdaEmb)
        coord = torch.einsum("bnm...,bmd->bnmd", U, LambdaEmb)  # (#graph, N, M, d)

        elvlncoord = self.elvln(coord)
        gs, gv, gv2 = self.gaggrs[0].forward(X, elvlncoord, nodemask)
        gv2 = gv2 * gsizenorm
        gv = gv * gsizenorm_v
        
        ts, tv = self.svmixs[0](self.sln(X), self.vln(coord))
        ts1, tv1 = self.tprods[0](ts, tv, gs, gv, gv2)
        
        coord = coord + tv1
        X = X + ts1.masked_fill(nodemask.unsqueeze(-1), 0)

        for i in range(1, self.num_layers):
            if self.elres:
                elvlncoord = self.elvln(coord)
                tgs, tgv, tgv2 = self.gaggrs[i](X, elvlncoord, nodemask)
                tgv = tgv * gsizenorm_v
                tgv2 = tgv2 * gsizenorm
                gs = gs + tgs
                gv = gv + tgv
                gv2 = gv2 + tgv2

            ts, tv = self.svmixs[i](self.sln(X), self.vln(coord))
            ts = ts.masked_fill(nodemask.unsqueeze(-1), 0)
            ts1, tv1 = self.tprods[i](ts, tv, gs, gv, gv2)

            coord = coord + tv1
            X = X + ts1.masked_fill(nodemask.unsqueeze(-1), 0)

        if self.nodetask:
            X = X
        else:
            X = self.pool(X, nodemask.unsqueeze(-1), 1)
        return self.predln(self.predlin(X))

    def forward(self, A, X, nodemask, *inputtuple):
        '''
        A (#graph, N, N)
        X (#graph, N, d)
        nodemask (#graph, N)
        '''
        pred = self.eigenforward(*self.inputencoder(A, X, nodemask, *inputtuple))
        if self.nodetask:
            pred = pred[torch.logical_not(nodemask)]
        return pred
