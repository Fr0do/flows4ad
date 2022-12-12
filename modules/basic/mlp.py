import torch.nn as nn

from typing import Union
from .utils import get_activation_class


class MultiLayerPerceptron(nn.Module):

    def __init__(
        self, 
        num_layers: int,
        d_in: int, 
        d_hidden: int, 
        d_out: int,
        activation: Union[nn.Module, str] = 'relu',
        activation_kwargs: dict = {},
        layer_norm: bool = False,
        layer_norm_kwargs: dict = {}
    ):
        super().__init__()
        assert num_layers >= 2

        if isinstance(activation, str):
            activation = get_activation_class(activation)(**activation_kwargs)

        layers = []
        for layer_id in range(num_layers):
            layer_d_in = (d_hidden, d_in)[layer_id == 0]
            layer_d_out = (d_hidden, d_out)[layer_id == num_layers - 1]
            layers.append(nn.Linear(layer_d_in, layer_d_out))
            if layer_id != num_layers - 1:
                layers.append(activation)
                if layer_norm:
                    layers.append(nn.LayerNorm(d_hidden, **layer_norm_kwargs))

        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net.forward(x)
