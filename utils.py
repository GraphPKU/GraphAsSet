import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import Sequential as PygSeq
from Norm import NoneNorm, normdict
import warnings

act_dict = {"relu": nn.ReLU(inplace=True), "ELU": nn.ELU(inplace=True), "silu": nn.SiLU(inplace=True), "softplus": nn.Softplus(), "softsign": nn.Softsign(), "softshrink": nn.Softshrink()}

class MLP(nn.Module):
    def __init__(self, indim: int, hiddim: int, outdim: int, numlayer: int=1, tailact: bool=True, dropout: float=0, norm: str="bn", activation: str="relu", tailbias=True, normparam: float=0.1) -> None:
        super().__init__()
        assert numlayer >= 0
        if isinstance(activation, str):
            activation = act_dict[activation]
        if isinstance(norm, str):
            norm = normdict[norm]
        if numlayer == 0:
            assert indim == outdim
            if norm != "none":
                warnings.warn("not equivalent to Identity")
                lin0 = nn.Sequential(norm(outdim, normparam))
            else:
                lin0 = nn.Sequential(NoneNorm())
        elif numlayer == 1:
            lin0 = nn.Sequential(nn.Linear(indim, outdim, bias=tailbias))
            if tailact:
                lin0.append(norm(outdim, normparam))
                if dropout > 0:
                    lin0.append(nn.Dropout(dropout, inplace=True))
                lin0.append(activation)
        else:
            lin0 = nn.Sequential(nn.Linear(hiddim, outdim, bias=tailbias))
            if tailact:
                lin0.append(norm(outdim, normparam))
                if dropout > 0:
                    lin0.append(nn.Dropout(dropout, inplace=True))
                lin0.append(activation)
            for _ in range(numlayer-2):
                lin0.insert(0, activation)
                if dropout > 0:
                    lin0.insert(0, nn.Dropout(dropout, inplace=True))
                lin0.insert(0, norm(hiddim, normparam))
                lin0.insert(0, nn.Linear(hiddim, hiddim))
            lin0.insert(0, activation)
            if dropout > 0:
                lin0.insert(0, nn.Dropout(dropout, inplace=True))
            lin0.insert(0, norm(hiddim, normparam))
            lin0.insert(0, nn.Linear(indim, hiddim))
        self.lin = lin0
        # self.reset_parameters()

    def forward(self, x: Tensor):
        return self.lin(x)


def freezeGNN(model: nn.Module):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
