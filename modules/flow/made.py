import os
import torch
import torch.nn as nn

from .masks import *
from .layers import MaskedLinear


class MADE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.init_layers()

    def init_layers():
        self.d = self.config.num_chunks or 2
        self.d_in = self.config.num_features
        self.d_hidden = self.config.hidden_dim
        self.ordering = np.arange(self.nin) # whether to permute inputs
        self.d_out = self.nin * d


        self.layers = []
        hs = [self.d_in] + self.d_hidden + [self.d_out]
        for h0, h1 in zip(hs, hs[1:]):
            self.layers.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        self.layers.pop() # remove last ReLU
        self.layers = nn.ModuleList(self.layers)

        self.m = {}
        self.create_mask()

    def create_mask(self):
        L = len(self.hidden_sizes)

        self.m[-1] = self.ordering
        for l in range(L):
            self.m[l] = np.random.randint(self.m[l - 1].min(),
                                          self.nin - 1, size=self.hidden_sizes[l])

        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        masks[-1] = np.repeat(masks[-1], self.d, axis=1)

        layers = [l for l in self.layers.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        batch_size = x.shape[0]
        out = x.view(batch_size, self.d_in)
        for layer in self.net:
            out = layer(out)
        out = out.view(batch_size, self.d_in, self.d)
        return out