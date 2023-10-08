from utils import MLP
import torch.nn as nn
import torch
import math
import numpy as np

import math


class SinEncoding(nn.Module):
    '''
    between (-1, 1)
    '''

    def __init__(self, dim=128, **kwargs):
        super().__init__()
        self.constant = 100
        self.dim = dim
        assert dim % 2 == 0
        self.register_buffer(
            "div", math.pi / (1 + torch.arange(dim // 2, dtype=torch.float)))

    def forward(self, e):
        # input:  (*)
        # output: (*,d)
        pe = e.unsqueeze(-1) * self.div
        ret = torch.cat((torch.sin(pe), torch.cos(pe)), dim=-1)
        return ret


class PolynomialEncoding(nn.Module):

    def __init__(self, dim, **kwargs) -> None:
        super().__init__()
        self.register_buffer("exponents", 1 + torch.arange(dim))

    def forward(self, e):
        # input:  (*)
        # output: (*,d)
        return torch.pow(e.unsqueeze(-1), self.exponents)


class MLPEncoding(nn.Module):

    def __init__(self, dim, activation, dropout=0.0, **kwargs):
        super(MLPEncoding, self).__init__()
        self.lin = MLP(1,
                       2 * dim,
                       dim,
                       dropout=dropout,
                       activation=activation,
                       **kwargs)

    def forward(self, e):
        # input:  (*)
        # output: (*,d)
        pe = e.unsqueeze(-1)
        ret = self.lin(pe)
        return ret
    
class IdEncoding(nn.Module):

    def __init__(self, dim, activation, dropout=0.0, **kwargs):
        super(IdEncoding, self).__init__()
        self.register_buffer("lin", torch.ones(dim))

    def forward(self, e):
        # input:  (*)
        # output: (*,d)
        pe = e.unsqueeze(-1) * self.lin
        return pe


class SincRadialBasis(nn.Module):
    # copied from Painn
    # (0, rbound upper)
    def __init__(self, num_rbf, rbound_upper, rbf_trainable=False, **kwargs):
        super().__init__()
        if rbf_trainable:
            self.register_parameter(
                "n",
                nn.parameter.Parameter(
                    torch.arange(1, num_rbf + 1,
                                 dtype=torch.float).unsqueeze(0) /
                    rbound_upper))
        else:
            self.register_buffer(
                "n",
                torch.arange(1, num_rbf + 1, dtype=torch.float).unsqueeze(0) /
                rbound_upper)

    def forward(self, r):
        n = self.n
        output = (math.pi) * n * torch.sinc(n * r)
        return output


class BesselBasisLayer(torch.nn.Module):
    # (0, rbound upper)
    def __init__(self,
                 num_rbf,
                 rbound_upper,
                 rbound_lower=0.0,
                 rbf_trainable=False,
                 **kwargs):
        super().__init__()
        freq = torch.arange(
            1, num_rbf + 1,
            dtype=torch.float).unsqueeze(0) * math.pi / rbound_upper
        if not rbf_trainable:
            self.register_buffer("freq", freq)
        else:
            self.register_parameter("freq", nn.parameter.Parameter(freq))

        self.rbound_upper = rbound_upper

    def forward(self, dist):
        '''
        dist (B, 1)
        '''
        return ((self.freq * dist).sin() /
                (dist + 1e-7)) * ((2 / self.rbound_upper)**0.5)


class GaussianSmearing(nn.Module):

    def __init__(self,
                 dim,
                 rbound_upper=-1,
                 rbound_lower=1,
                 rbf_trainable=False,
                 **kwargs):
        super(GaussianSmearing, self).__init__()
        self.rbound_lower = rbound_lower
        self.rbound_upper = rbound_upper
        self.num_rbf = dim
        self.rbf_trainable = rbf_trainable

        offset, coeff = self._initial_params()
        if rbf_trainable:
            self.register_parameter("coeff", nn.parameter.Parameter(coeff))
            self.register_parameter("offset", nn.parameter.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(self.rbound_lower, self.rbound_upper,
                                self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0])**2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        return torch.exp(self.coeff *
                         torch.square(dist.unsqueeze(-1) - self.offset))


class GaussianGaussian(nn.Module):
    '''
    [-1, 1]
    '''

    def __init__(self, dim, rbf_trainable=False, **kwargs):
        super().__init__()
        self.num_rbf = dim
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()
        if rbf_trainable:
            self.register_parameter("means", nn.parameter.Parameter(means))
            self.register_parameter("coeff", nn.parameter.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("coeff", betas)

    def _initial_params(self):
        means = torch.special.erfinv(
            torch.linspace(-0.999, 0.999, self.num_rbf))
        means /= torch.max(means)
        delta = torch.zeros_like(means)
        diff = torch.diff(means)
        delta[1:] += diff
        delta[:-1] += diff
        delta[1:-1] /= 2
        coeff = -0.5 / (delta)**2
        return means, coeff

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.coeff.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return torch.exp(self.coeff * torch.square(dist - self.means))


rbf_class_mapping = {
    "gauss": GaussianSmearing,
    "gg": GaussianGaussian,
    "mlp": MLPEncoding,
    "id": IdEncoding,
    "sin": SinEncoding
}
