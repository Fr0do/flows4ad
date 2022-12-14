import os
import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from typing import List, Tuple, Optional
from numpy.random import permutation, randint

from .masks import *
from ..basic import MultiLayerPerceptron, get_activation_class


class AffineCouplingLayer(nn.Module):

    def __init__(
        self, 
        d_embed: int, 
        d_hidden: int,
        mask_type: str = MaskType.CHANNEL_WISE, 
        invert_mask: bool = False,
        num_mlp_layers: int = 2,
        mlp_activation: str = 'relu',
        mlp_activation_kwargs: dict = {},
        layer_norm: bool = False,
        layer_norm_kwargs: dict = {}
    ):
        super().__init__()
        # Save mask info
        self.mask_type = mask_type
        self.invert_mask = invert_mask
        # learnable scale and shift function
        self.shift_scale = MultiLayerPerceptron(
            num_layers=num_mlp_layers,
            d_in=d_embed,
            d_hidden=d_hidden,
            d_out=2 * d_embed,
            activation=mlp_activation,
            activation_kwargs=mlp_activation_kwargs,
            layer_norm=layer_norm,
            layer_norm_kwargs=layer_norm_kwargs
        )

    def forward(self, x, reverse=False):
        # make mask
        if self.mask_type == MaskType.CHECKERBOARD:
            mask = make_checkerboard_mask(x, self.invert_mask)
        elif self.mask_type == MaskType.CHANNEL_WISE:
            mask = make_channel_wise_mask(x, self.invert_mask)
        x_mask = x * mask
        shift, scale = self.shift_scale(x_mask).chunk(2, dim=-1)
        
        shift = shift * ~mask
        scale = scale * ~mask

        if reverse:
            x = (x - shift) * torch.exp(-scale)
        else:
            x = x * torch.exp(scale) + shift

        # the output is transformed input 
        # and logarithm of jacobian (which equals to s)
        return x, scale

# adapted from https://github.com/e-hulten/maf/blob/master/maf_layer.py

class MaskedLinear(nn.Linear):
    """Linear transformation with masked out elements. y = x.dot(mask*W.T) + b"""

    def __init__(self, d_in: int, d_out: int, bias: bool = True) -> None:
        """
        Args:
            d_in: Size of each input sample.
            d_out:Size of each output sample.
            bias: Whether to include additive bias. Default: True.
        """
        super().__init__(d_in, d_out, bias)
        self.register_buffer('mask', torch.zeros_like(self.weight))

    def initialise_mask(self, mask: Tensor):
        """Internal method to initialise mask."""
        self.mask = mask

    def forward(self, x: Tensor) -> Tensor:
        """Apply masked linear transformation."""
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):

    def __init__(
        self,
        d_in: int,
        hidden_dims: List[int],
        gaussian: bool = False,
        random_order: bool = False,
        seed: Optional[int] = None,
        activation: str = 'relu',
        activation_kwargs: dict = {},
    ):
        """Initalise MADE model.
    
        Args:
            d_in: Size of input.
            hidden_dims: List with sizes of the hidden layers.
            gaussian: Whether to use Gaussian MADE. Default: False.
            random_order: Whether to use random order. Default: False.
            seed: Random seed for numpy. Default: None.
        """
        super().__init__()
        # Set random seed.
        np.random.seed(seed)
        self.d_in = d_in
        self.d_out = 2 * d_in if gaussian else d_in
        self.hidden_dims = hidden_dims
        self.random_order = random_order
        self.gaussian = gaussian
        self.masks = {}
        self.mask_matrix = []
        self.layers = []

        if isinstance(activation, str):
            activation = get_activation_class(activation)(**activation_kwargs)

        # List of layers sizes.
        dim_list = [self.d_in, *hidden_dims, self.d_out]
        # Make layers and activation functions.
        for i in range(len(dim_list) - 2):
            self.layers.append(MaskedLinear(dim_list[i], dim_list[i + 1]),)
            self.layers.append(activation)
        # Hidden layer to output layer.
        self.layers.append(MaskedLinear(dim_list[-2], dim_list[-1]))
        # Create model.
        self.model = nn.Sequential(*self.layers)
        # Get masks for the masked activations.
        self._create_masks()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if self.gaussian:
            # If the output is Gaussan, return raw mus and sigmas.
            return self.model(x)
        else:
            # If the output is Bernoulli, run it trough sigmoid to squash p into (0,1).
            return torch.sigmoid(self.model(x))

    def _create_masks(self) -> None:
        """Create masks for the hidden layers."""
        # Define some constants for brevity.
        L = len(self.hidden_dims)
        D = self.d_in

        # Whether to use random or natural ordering of the inputs.
        self.masks[0] = permutation(D) if self.random_order else np.arange(D)

        # Set the connectivity number m for the hidden layers.
        # m ~ DiscreteUniform[min_{prev_layer}(m), D-1]
        for l in range(L):
            low = self.masks[l].min()
            size = self.hidden_dims[l]
            self.masks[l + 1] = randint(low=low, high=D - 1, size=size)

        # Add m for output layer. Output order same as input order.
        self.masks[L + 1] = self.masks[0]

        # Create mask matrix for input -> hidden_1 -> ... -> hidden_L.
        for i in range(len(self.masks) - 1):
            m = self.masks[i]
            m_next = self.masks[i + 1]
            # Initialise mask matrix.
            M = torch.zeros(len(m_next), len(m))
            for j in range(len(m_next)):
                # Use broadcasting to compare m_next[j] to each element in m.
                M[j, :] = torch.from_numpy((m_next[j] >= m).astype(int))
            # Append to mask matrix list.
            self.mask_matrix.append(M)

        # If the output is Gaussian, double the number of output units (mu,sigma).
        # Pairwise identical masks.
        if self.gaussian:
            m = self.mask_matrix.pop(-1)
            self.mask_matrix.append(torch.cat((m, m), dim=0))

        # Initalise the MaskedLinear layers with weights.
        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(next(mask_iter))


class MAFLayer(nn.Module):

    def __init__(
        self, 
        dim: int, 
        hidden_dims: List[int], 
        inverse: bool,
        activation: str = 'relu',
        activation_kwargs: dict = {},
    ):
        """
        Args:
            dim: Dimension of input. E.g.: dim = 784 when using MNIST.
            hidden_dims: List with of sizes of the hidden layers in each MADE. 
            inverse: Whether to reverse the input vector in each MADE. 
        """
        super().__init__()
        self.dim = dim
        self.made = MADE(
            dim, 
            hidden_dims, 
            gaussian=True, 
            seed=None, 
            activation=activation, 
            activation_kwargs=activation_kwargs
        )
        self.inverse = inverse

    def forward(self, x: Tensor, reverse=False) -> Tuple[Tensor, Tensor]:
        if reverse:
            out = self.made(x.float())
            mu, logp = torch.chunk(out, 2, dim=1)
            u = (x - mu) * torch.exp(0.5 * logp)
            u = u.flip(dims=(1,)) if self.inverse else u
            log_det = 0.5 * torch.sum(logp, dim=1, keepdims=True)
        else:
            # will fail:(
            x = x.flip(dims=(1,)) if self.inverse else x
            u = torch.zeros_like(x)
            for dim in range(self.dim):
                out = self.made(u)
                mu, logp = torch.chunk(out, 2, dim=1)
                mod_logp = torch.clamp(-0.5 * logp, max=10)
                u[:, dim] = mu[:, dim] + x[:, dim] * torch.exp(mod_logp[:, dim])
            log_det = torch.sum(mod_logp, axis=1, keepdims=True)
        return u, log_det
