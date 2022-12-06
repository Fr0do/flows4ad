import torch
import torch.nn as nn

from .flow import GeneralFlow
from .layers import AffineCouplingLayer

from .masks import *
from .invertible import InvertibleLinear

from flows4ad.modules.basic import MovingBatchNorm1d


class Glow(GeneralFlow):
    '''
    Identical to RealNVP except addtion of InvertibleLinear and MovingBatchNorm1d
    '''

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.init_layers()
    
    def init_layers(self):
        d_embed = self.config.num_features
        d_hidden = self.config.hidden_dim

        num_mlp_layers = self.config.num_mlp_layers
        num_flow_layers = self.config.num_flow_layers
        
        use_channel_wise_splits = self.config.use_channel_wise_splits
        use_checkerboard_splits = self.config.use_checkerboard_splits
        
        mlp_activation = self.config.mlp_activation
        use_layer_norm = self.config.use_layer_norm
        use_batch_norm = self.config.use_batch_norm

        mlp_activation_kwargs = {}
        layer_norm_kwargs = {}
        batch_norm_kwargs = {}


        assert num_flow_layers % 2 == 0, \
            "Number of flow layers should be even"
        assert use_channel_wise_splits or use_checkerboard_splits, \
            "Use at least one of split strategies."

       
        mask_types = []
        if use_channel_wise_splits:
            mask_types.append(MaskType.CHANNEL_WISE)
        if use_checkerboard_splits:
            mask_types.append(MaskType.CHECKERBOARD)

        layers = []
        for i in range(num_flow_layers):
            invert_mask = i % 2 == 0
            mask_type = (i // 2) % len(mask_types)
            layers.append(InvertibleLinear(d_embed))
            layers.append(
                AffineCouplingLayer(
                    d_embed=d_embed,
                    d_hidden=d_hidden,
                    mask_type=mask_type,
                    invert_mask=invert_mask,
                    num_mlp_layers=num_mlp_layers,
                    mlp_activation=mlp_activation,
                    mlp_activation_kwargs=mlp_activation_kwargs,
                    layer_norm=use_layer_norm,
                    layer_norm_kwargs=layer_norm_kwargs
                )
            )
            if use_batch_norm:
                layers.append(MovingBatchNorm1d(d_embed, **batch_norm_kwargs))

        self.layers = nn.Sequential(*layers)
