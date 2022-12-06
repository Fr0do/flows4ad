import torch
import torch.nn as nn


class NoneEmbedding(nn.Module):
    def get_embedding_size(embedding_config):
        return embedding_config.num_features
        
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, x):
        return x
