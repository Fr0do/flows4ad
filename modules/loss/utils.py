from .density import NormalizingFlowLoss


def get_loss(prior, config=None):
    loss = NormalizingFlowLoss(prior)
    return loss