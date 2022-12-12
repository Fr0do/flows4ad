import torch
import torch.nn as nn

from ..basic import MultiLayerPerceptron


__all__ = ["VariationalAutoEncoder"]


class VariationalAutoEncoder(nn.Module):

    def __init__(self, encoder_config) -> None:
        super().__init__()
        self.config = encoder_config
        self.init_layers()

    def init_layers(self):
        num_encoder_layers = self.config.num_encoder_layers
        num_decoder_layers = self.config.num_decoder_layers
        
        d_in = self.config.num_features
        d_hidden = self.config.hidden_dim
        d_latent = self.config.latent_dim

        activation = getattr(self.config, 'activation', 'relu')
        activation_kwargs = getattr(self.config, 'activation_kwargs', {})
        
        layer_norm = getattr(self.config, 'use_layer_norm', False)
        layer_norm_kwargs = getattr(self.config, 'layer_norm_kwargs', {})

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