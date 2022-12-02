from . import flow


def get_loss(prior, config=None):
    loss = flow.RealNVPLoss(prior)
    return loss