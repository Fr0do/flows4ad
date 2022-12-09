import torch
import torch.nn as nn

from typing import Union
from ..basic.mlp import MultiLayerPerceptron


__all__ = ["VariationalAutoEncoder"]


class VariationalAutoEncoder(nn.Module):

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_in: int,
        d_hidden: int, 
        d_latent: int,
        activation: Union[nn.Module, str] = 'relu',
        activation_kwargs: dict = {},
        layer_norm: bool = False,
        layer_norm_kwargs: dict = {}
    ) -> None:
        super().__init__()

        self.encoder = MultiLayerPerceptron(
            num_encoder_layers, 
            d_in, 
            d_hidden, 
            d_latent, 
            activation, 
            activation_kwargs,
            layer_norm,
            layer_norm_kwargs
        )

        self.mu = nn.Linear(d_latent, d_latent)
        self.log_sigma = nn.Linear(d_latent, d_latent)

        self.decoder = MultiLayerPerceptron(
            num_decoder_layers, 
            d_latent, 
            d_hidden, 
            d_in, 
            activation, 
            activation_kwargs,
            layer_norm,
            layer_norm_kwargs
        )

    @classmethod
    def from_config(cls, config):
        return cls(
            config.num_encoder_layers,
            config.num_decoder_layers,
            config.d_in,
            config.d_hidden,
            config.d_latent,
            getattr(config, 'activation', 'relu'),
            getattr(config, 'activation_kwargs', {}),
            getattr(config, 'layer_norm', False),
            getattr(config, 'layer_norm_kwargs', {})
        )

    def encode(self, x):
        x = self.encoder(x)
        mu, log_sigma = self.mu(x), self.log_sigma(x)
        return mu, log_sigma

    def reparametrize(self, mu, log_sigma):
        return mu + log_sigma.exp() * torch.randn_like(log_sigma)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparametrize(mu, log_sigma)
        return self.decode(z), mu, log_sigma