import torch
import torch.nn as nn


class GeneralFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = None

    def init_layers(self):
        raise NotImplementedError()
        
    def forward(self, x, reverse=False):
        log_det = 0.0
        layers = reversed(self.layers) if reverse else self.layers
        for flow_layer in layers:
            x, scale = flow_layer(x, reverse)
            log_det += scale
        return x, log_det