import torch
import torch.nn as nn

from ..embedding import get_embedding_instace
from ..flow import get_flow_instace


class FlowDetector(nn.Module):
    
    def __init__(self, detector_config):
        super().__init__()
        self.config = detector_config

        self.embedding = get_embedding_instace(self.config.embedding_config)
        self.flow = get_flow_instace(self.config.flow_config)
    
    def forward(self, x, **kwargs):
        x_embedded = self.embedding(x)
        x_transformed, log_det = self.flow(x_embedded, **kwargs)

        return x_transformed, log_det
