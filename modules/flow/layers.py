import os
import torch
import torch.nn as nn

from .masks import *
from flows4ad.modules.basic import MultiLayerPerceptron


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