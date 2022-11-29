import torch.nn as nn
import torch.nn.functional as F


from typing import Union


__all__ = [
    "ResidualBlock"
    "ResNet"
]


class ResidualBlock(nn.Module):
    """ResNet basic block with weight norm."""
    def __init__(
        self, 
        d_embed: int, 
        d_hidden: int = None, 
        activation: Union[nn.Module, str] = 'ReLU',
        activation_kwargs: dict = {}  
    ):
        super(ResidualBlock, self).__init__()

        if isinstance(activation, str):
            activation = getattr(nn, activation)(**activation_kwargs) 
        self.activation = activation

        if d_hidden is None:
            d_hidden = d_embed       

        self.in_norm = nn.BatchNorm2d(d_embed)
        self.in_fc = nn.Linear(d_embed, d_hidden, bias=False)

        self.out_norm = nn.BatchNorm2d(d_hidden)
        self.out_fc = nn.Linear(d_hidden, d_embed, bias=True)

    def forward(self, x):
        skip = x

        x = self.in_norm(x)
        x = self.self.activation(x)
        x = self.in_fc(x)

        x = self.out_norm(x)
        x = self.self.activation(x)
        x = self.out_fc(x)

        x = x + skip

        return x

# TODO maybe not needed
class ResNet:
    pass
