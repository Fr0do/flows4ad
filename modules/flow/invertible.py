import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "InvertibleLinear"
]

class InvertibleLinear(nn.Module):

    def __init__(self, d_embed: int):
        super(InvertibleLinear, self).__init__()

        weight_init, _ = torch.linalg.qr(torch.randn(d_embed, d_embed))
        # P, L, U = torch.lu_unpack(*torch.linalg.lu_factor(weight_init))
        P, L, U = torch.lu_unpack(*torch.lu(weight_init))
        s = torch.diag(U)
        sign_s = torch.sign(s)
        log_s = torch.log(torch.abs(s))
        U = torch.triu(U, diagonal=1)
        # register buffer and parameters
        self.register_buffer("P", P)
        self.register_buffer("sign_s", sign_s)
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)
        self.log_s = nn.Parameter(log_s)        
        # prevent gradient flow
        self.L.register_hook(
            lambda grad: grad * torch.tril(torch.ones_like(grad), diagonal=-1)
        )
        self.U.register_hook(
            lambda grad: grad * torch.triu(torch.ones_like(grad), diagonal=+1)
        )

    @property
    def weight(self):
        U = self.U + torch.diag(self.sign_s * self.log_s.exp())
        return self.P @ self.L @ U

    def forward(self, x, reverse: bool = False):
        if not reverse:
            x = F.linear(x, self.weight)
        else:
            x = F.linear(x,  torch.inverse(self.weight))
        return x, self.log_s.repeat(x.shape[0], 1)
