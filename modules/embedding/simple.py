import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEmbedding(nn.Module):
    
    def get_embedding_size(embedding_config):
        return embedding_config.num_features * embedding_config.num_dimensions

    def __init__(self, embedding_config):
        super().__init__()
        self.config = embedding_config
        self.init_layers()

    def init_layers(self):
        num_features = self.config.num_features

        num_dimensions = self.config.num_dimensions
        variance = self.config.variance

        initialisation = self.config.initialisation
        trainable = self.config.trainable
        
        if initialisation == 'normal':
            coefficient_values = torch.normal(0.0, variance, (num_features, num_dimensions))
        else:
            raise ValueError(f'No such initialisation option as {initialisation}!')
        
        if trainable:
            self.coefficients = nn.Parameter(coefficient_values)
        else:
            self.register_buffer('coefficients', coefficient_values)

    def forward(self, x):
        z = F.relu(x[..., None] * self.coefficients[None])
        z = z.reshape(x.shape[0], -1)
        return z