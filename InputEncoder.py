import torch
import torch.nn as nn
from Emb import MultiEmbedding
from utils import MLP
from basis_layers import rbf_class_mapping
from typing import Final

EPS = 1e-4


class QInputEncoder(nn.Module):
    LambdaBound: Final[float]
    laplacian: Final[bool]

    def __init__(self, featdim, hiddim, LambdaBound=1e-4, **kwargs) -> None:
        super().__init__()
        self.LambdaBound = LambdaBound
        self.featdim = featdim
        if kwargs["dataset"] in ["pepfunc", "pepstruct"]:
            self.xemb = MultiEmbedding(hiddim, [18, 4, 8, 8, 6, 2, 7, 3, 3],
                                       **kwargs["xemb"])
            self.edgeEmb = MultiEmbedding(hiddim, [5, 5, 5], **kwargs["xemb"])
        elif kwargs["dataset"].startswith("qm9"):
            self.xemb = nn.Linear(11, hiddim)
            self.edgeEmb = nn.Linear(4, hiddim)
        elif kwargs["dataset"] == "ogbg-molhiv":
            self.xemb = MultiEmbedding(hiddim, kwargs["xembdims"],
                                       **kwargs["xemb"])
            tmp = kwargs["xemb"].copy()
            self.tedgeEmb = MultiEmbedding(featdim, [100, 100, 100], **tmp)
            self.edgeEmb = lambda x: self.tedgeEmb(x.to(torch.long))
        elif kwargs["dataset"] in ["zinc", "zinc-full"]:
            self.xemb = MultiEmbedding(hiddim, [40], **kwargs["xemb"])
            self.edgeEmb = MultiEmbedding(hiddim, [20], **kwargs["xemb"])
        elif kwargs["dataset"] == "pascalvocsp":
            self.xemb = nn.Sequential(nn.Linear(14, hiddim))
            self.edgeEmb = nn.Sequential(nn.Linear(2, hiddim))
        else:
            raise NotImplementedError
        self.LambdaEmb = rbf_class_mapping[kwargs["lexp"]](
            hiddim, **kwargs["basic"], **kwargs["lambdaemb"])
        self.degreeEmb = MultiEmbedding(
            hiddim, [100], **kwargs["xemb"]) if kwargs["degreeemb"] else None
        self.distEmb = rbf_class_mapping[kwargs["lexp"]](hiddim,
                                                         **kwargs["basic"],
                                                         **kwargs["lambdaemb"])
        self.normA = kwargs["normA"]
        self.laplacian = kwargs["laplacian"]
        self.decompnoise = kwargs["decompnoise"]
        self.use_pos = kwargs["use_pos"]

    def setnoiseratio(self, ratio):
        self.decompnoise = ratio

    def forward(self, A, X, nodemask, pos=None):
        '''
        A (b, n, n, d)
        '''
        eA = self.edgeEmb(A)
        A = torch.any(A != 0, dim=-1).to(torch.float)
        D = torch.sum(A, dim=-1)  # (#graph, N)
        if self.laplacian:
            L = torch.diag_embed(D) - A
        else:
            L = A  # (#graph, N, N)
        if self.normA:
            tD = torch.clamp_min(D, 1)  # (# graph, N, N)
            tD = torch.rsqrt_(tD)
            L = tD.unsqueeze(1) * L * tD.unsqueeze(2)
        if self.training:
            N = L.shape[1]
            perm = torch.randperm(N, device=L.device)
            L = L[:, perm][:, :, perm]
            invperm = torch.empty_like(perm)
            invperm[perm] = torch.arange(N, device=perm.device)
            Lambda, U = torch.linalg.eigh(L)
            U = U[:, invperm]  # (#graph, N, M)
        else:
            Lambda, U = torch.linalg.eigh(L)
        if self.laplacian:
            Lambda = Lambda[:, 1:]
            U = U[:, :, 1:]
        X = self.xemb(X)
        if self.degreeEmb is not None:
            X *= self.degreeEmb(D.to(torch.long).unsqueeze(-1))
        X.masked_fill_(nodemask.unsqueeze(-1), 0)
        Lambdamask = torch.abs(
            Lambda) < self.LambdaBound  # (#graph, M) # mask zero frequency
        U.masked_fill_(Lambdamask.unsqueeze(1), 0)
        U.masked_fill_(nodemask.unsqueeze(-1), 0)
        if self.laplacian:
            Lambda = torch.sqrt(torch.relu_(Lambda))
        if self.training:
            Lambda += self.decompnoise * torch.randn_like(Lambda)
            U += self.decompnoise * torch.randn_like(U)
        LambdaEmb = self.LambdaEmb(Lambda)  # (#graph, M, d2)
        if self.use_pos:
            dist = torch.norm(pos.unsqueeze(1) - pos.unsqueeze(2), dim=-1)
            dist = self.distEmb(dist)
            eA = eA * dist
        eA.masked_fill_(nodemask.unsqueeze(1).unsqueeze(-1), 0)
        eA.masked_fill_(nodemask.unsqueeze(2).unsqueeze(-1), 0)
        U = torch.einsum("bnmd,bml->bnld", eA, U)
        return LambdaEmb, Lambdamask, U, X, nodemask
