import torch
import torch.nn as nn

from ..autoencoder import get_autoencoder_instace


encoder_type_to_function = {
    'autoenccoder': get_autoencoder_instace
    # 'supervised_encoder': get_supervised_encoder
}


class TrainableEncoder(nn.Module):
    def __init__(self, encoder_config):
        super().__init__()
        self.config = encoder_config
        self.model = encoder_type_to_function[self.config.encoder_type](self.config)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)