import torch
from typing import Final
import torch.nn as nn
from MaskedReduce import reduce_dict
from InputEncoder import QInputEncoder
import PermEquiLayer
from utils import MLP
import torch.nn.functional as F


class DirCFConv(nn.Module):

    def __init__(self, hiddim, uselinv=True, **kwargs):
        super().__init__()
        self.lins = MLP(hiddim, hiddim, hiddim, **kwargs)
        self.linv = nn.Linear(hiddim, hiddim, bias=False) if uselinv else nn.Identity()

    def forward(self, s, v, el):
        '''
        x (B, N, d)
        v (B, N, M, d)
        el (B, N, N, d)
        '''
        s = torch.einsum("bijd,bjd->bid", el, self.lins(s))
        v = torch.einsum("bijd,bjmd->bimd", el, self.linv(v))
        return s, v


class svMix(nn.Module):

    res: Final[bool]

    def __init__(self, hiddim, uselinv=True, res=True, boostsv=False, **kwargs) -> None:
        super().__init__()
        self.linv1 = nn.Linear(hiddim, hiddim, bias=False) if uselinv else nn.Identity()
        self.linv2 = nn.Linear(hiddim, hiddim, bias=False) if uselinv else nn.Identity()
        self.linv3 = nn.Linear(hiddim, hiddim, bias=False) if boostsv and uselinv else nn.Identity()
        self.lins1 = MLP(hiddim, hiddim, hiddim, **kwargs)
        self.lins2 = MLP(hiddim, hiddim, hiddim, **kwargs)
        self.lins3 = MLP(hiddim, hiddim, hiddim, **kwargs) if boostsv else nn.Identity()
        self.res = res

    def forward(self, s, v):
        '''
        s (B, N, d)
        v (B, N, M, d)
        keep zero
        '''
        vprod = self.lins3(torch.einsum("bnmd,bnmd->bnd", self.linv1(v), self.linv2(v)))
        if self.res:
            ts = s + self.lins1(s) * vprod  # (B, N, d)
            tv = v + torch.einsum("bnmd,bnd->bnmd", self.linv3(v), self.lins2(s))
        else:
            ts = self.lins1(s) * vprod  # (B, N, d)
            tv = torch.einsum("bnmd,bnd->bnmd", self.linv3(v), self.lins2(s))
        return ts, tv


class sv2el(nn.Module):
    uses: Final[bool]
    def __init__(self, indim, hiddim, uselinv=True, uselins=True, uses=True, **kwargs) -> None:
        super().__init__()
        self.linv1 = nn.Linear(indim, hiddim, bias=False) if (uselinv or indim!=hiddim) else nn.Identity()
        self.linv2 = nn.Linear(indim, hiddim, bias=False) if (uselinv or indim!=hiddim) else nn.Identity()
        self.lins1 = nn.Linear(indim, hiddim) if (uselins or indim!=hiddim)  else nn.Identity()
        self.lins2 = nn.Linear(indim, hiddim) if (uselins or indim!=hiddim) else nn.Identity()
        self.lin = MLP(hiddim, hiddim, hiddim, **kwargs)
        self.uses = uses

    def forward(self, s, v1, v0):
        '''
        s (b, n, d)
        v (b, n, m, d)
        '''
        if self.uses:
            ret = self.lin(
                torch.einsum("bid,bjd,bimd,bjmd->bijd",
                            self.lins1(s), self.lins2(s), self.linv1(v1),
                            self.linv2(v0))) 
        else:
            ret = self.lin(torch.einsum("bimd,bjmd->bijd",
                            self.linv1(v1), self.linv2(v0))) 
        # print(torch.linalg.norm(v1).item(), torch.linalg.norm(ret).item(), torch.linalg.norm(self.linv1.weight).item(), torch.linalg.norm(self.linv2.weight).item())
        return ret

class VNorm(nn.Module):

    def __init__(self, hiddim, elementwise_affine: bool=False) -> None:
        super().__init__()
        assert not elementwise_affine

    def forward(self, v):
        '''
        v (*, m, d)
        '''
        v = F.normalize(v, dim=-2, eps=1e-3)
        return v
    

class VMean(nn.Module):

    def __init__(self, hiddim, elementwise_affine: bool=False) -> None:
        super().__init__()
        assert not elementwise_affine

    def forward(self, v):
        '''
        v (*, m, d)
        '''
        v = v - torch.mean(v, dim=-1, keepdim=True)
        return v
    
class Imod(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, *args):
        return args

class PiOModel(nn.Module):
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
        self.LambdaEncoder = PermEquiLayer.PermEquiLayer(hiddim, hiddim, "deepset",
                                           False, **kwargs["l_model"])

        self.elprojs = nn.ModuleList(
            [sv2el(hiddim, caldim, **kwargs["elproj"]) for _ in range(num_layers)]
        )
        
        self.svmixs = nn.ModuleList(
            [svMix(hiddim, **kwargs["svmix"]) if usesvmix else Imod() for _ in range(num_layers)]
        )
        
        self.convs = nn.ModuleList(
            [DirCFConv(hiddim, **kwargs["conv"]) for _ in range(num_layers)]
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
        LambdaEmb = self.LambdaEncoder(LambdaEmb, LambdaMask)  # LambdaEmb (#graph, M, d1)
        LambdaEmb = torch.where(LambdaMask.unsqueeze(-1), 0, LambdaEmb)
        coord = torch.einsum("bnm...,bmd->bnmd", U, LambdaEmb)  # (#graph, N, M, d)
        nnfilter = torch.logical_not(torch.logical_or(nodemask.unsqueeze(-1), nodemask.unsqueeze(1))).float().unsqueeze(-1)
        elvlncoord = self.elvln(coord)
        el = self.elprojs[0](X, elvlncoord, elvlncoord) * (gsizenorm * nnfilter)# + A
        ts, tv = self.svmixs[0](self.sln(X), self.vln(coord))
        ts1, tv1 = self.convs[0](ts, tv, el)
        coord = coord + tv1
        X = X + ts1
        for i in range(1, self.num_layers):
            if self.elres:
                elvlncoord = self.elvln(coord)
                el = el + nnfilter*self.elprojs[i](X, elvlncoord, elvlncoord) * (gsizenorm * nnfilter)
            ts, tv = self.svmixs[i](self.sln(X), self.vln(coord))
            ts1, tv1 = self.convs[i](ts, tv, el)
            coord = coord + tv1
            X = X + ts1
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
