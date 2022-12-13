from typing import List, Optional
import numpy as np
from numpy.random import permutation, randint
import torch
import torch.nn as nn
from torch import Tensor

from .flow import GeneralFlow
from .layers import MAFLayer

from ..basic import MovingBatchNorm1d


class MAF(GeneralFlow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_layers()

    def init_layers(self):
        d_embed = self.config.num_features
        d_hidden = self.config.hidden_dim
        is_inverse_made = self.config.is_inverse_made
        num_flow_layers = self.config.num_flow_layers
        num_mlp_layers = self.config.num_mlp_layers
        use_batch_norm = self.config.use_batch_norm
        batch_norm_kwargs = {}

        layers = []
        for _ in range(num_flow_layers):
            layers.append(MAFLayer(d_embed, 
                               num_mlp_layers * [d_hidden], 
                               inverse=is_inverse_made))
            if use_batch_norm:
                layers.append(MovingBatchNorm1d(d_embed, **batch_norm_kwargs))
        self.layers = nn.Sequential(*layers)
