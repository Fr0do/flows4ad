import os
import torch
import torch.nn as nn
from torch import Tensor
from .flow import GeneralFlow

class PlanarTransform(nn.Module):
    """Implementation of the invertible transformation used in planar flow:
        g(z) = z + u * h(dot(w.T, z) + b)
    See Section 4.1 in https://arxiv.org/pdf/1505.05770.pdf. 
    """

    def __init__(self, dim: int):
        """Initialise weights and bias.
        
        Args:
            dim: Dimensionality of the distribution to be estimated.
        """
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, dim).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(1).normal_(0, 0.1))
        self.u = nn.Parameter(torch.randn(1, dim).normal_(0, 0.1))

    def forward(self, z: Tensor, reverse=True):
        if torch.mm(self.u, self.w.T) < -1:
            self.get_u_hat()
        x = z + self.u * nn.Tanh()(torch.mm(z, self.w.T) + self.b)
        log_det = self.log_det(z)
        return x, log_det

    def log_det(self, z: Tensor):
        if torch.mm(self.u, self.w.T) < -1:
            self.get_u_hat()
        a = torch.mm(z, self.w.T) + self.b
        psi = (1 - nn.Tanh()(a) ** 2) * self.w
        abs_det = (1 + torch.mm(self.u, psi.T)).abs()
        log_det = torch.log(1e-4 + abs_det)
        return log_det.view(z.shape[0], 1)

    def get_u_hat(self):
        """Enforce w^T u >= -1. When using h(.) = tanh(.), this is a sufficient condition 
        for invertibility of the transformation f(z). See Appendix A.1. 
        of https://arxiv.org/pdf/1505.05770.pdf
        """
        wtu = torch.mm(self.u, self.w.T)
        m_wtu = -1 + torch.log(1 + torch.exp(wtu))
        self.u.data = (
            self.u + (m_wtu - wtu) * self.w / torch.norm(self.w, p=2, dim=1) ** 2
        )

class PlanarFlow(GeneralFlow):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.init_layers()

    def init_layers(self):
        d_embed = self.config.num_features
        num_flow_layers = self.config.num_flow_layers

        layers = [PlanarTransform(d_embed) for _ in range(num_flow_layers)]
        self.layers = nn.Sequential(*layers)
