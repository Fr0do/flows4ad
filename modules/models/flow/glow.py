import torch
import torch.nn as nn

from torch import Tensor

from .masks import *
from .inv_linear import InvertibleLinear
from .real_nvp import AffineCouplingLayer
from ..normalization import MovingBatchNorm1d


__all__ = [
    "Glow"
]


class Glow(nn.Module):
    '''
    Identical to RealNVP except addtion of InvertibleLinear and MovingBatchNorm1d
    '''

    def __init__(
        self,
        d_embed: int, 
        d_hidden: int,
        num_flow_layers: int,
        use_channel_wise_splits: bool = True,
        use_checkerboard_splits: bool = True,
        num_mlp_layers: int = 2,
        mlp_activation: str = 'relu',
        mlp_activation_kwargs: dict = {},
        layer_norm: bool = False,
        layer_norm_kwargs: dict = {},
        batch_norm: bool = False,
        batch_norm_kwargs: dict = {},
    ):
        super().__init__()

        assert num_flow_layers % 2 == 0, \
            "Number of flow layers should be even"
        assert use_channel_wise_splits or use_checkerboard_splits, \
            "Use at least one of split strategies."
       
        mask_types = []
        if use_channel_wise_splits:
            mask_types.append(MaskType.CHANNEL_WISE)
        if use_checkerboard_splits:
            mask_types.append(MaskType.CHECKERBOARD)

        flow_layers = []
        for i in range(num_flow_layers):
            invert_mask = i % 2 == 0
            mask_type = (i // 2) % len(mask_types)
            flow_layers.append(InvertibleLinear(d_embed))
            flow_layers.append(
                AffineCouplingLayer(
                    d_embed=d_embed,
                    d_hidden=d_hidden,
                    mask_type=mask_type,
                    invert_mask=invert_mask,
                    num_mlp_layers=num_mlp_layers,
                    mlp_activation=mlp_activation,
                    mlp_activation_kwargs=mlp_activation_kwargs,
                    layer_norm=layer_norm,
                    layer_norm_kwargs=layer_norm_kwargs
                )
            )
            if batch_norm:
                flow_layers.append(MovingBatchNorm1d(d_embed, **batch_norm_kwargs))

        self.flow_layers = nn.Sequential(*flow_layers)

        
    def forward(self, x: Tensor, reverse: bool = False):
        log_det = 0.0
        flow_layers = reversed(self.flow_layers) if reverse else self.flow_layers
        for flow_layer in flow_layers:
            x, scale = flow_layer(x, reverse)
            log_det += scale
        return x, log_det
