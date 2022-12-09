from .density import NormalizingFlowLoss
from .vae import VAELoss


def get_loss(prior, config=None):
    loss = NormalizingFlowLoss(prior)
    return loss

def get_vae_loss(config=None):
    loss = VAELoss()
    return loss
