import torch.nn as nn


__all__ = [
    "NormalizingFlowLoss"
]


class NormalizingFlowLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.
    Args:
        prior 
    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, prior):
        super(NormalizingFlowLoss, self).__init__()
        self.prior = prior

    def forward(self, z, log_det):
        log_prob = self.prior.log_prob(z) + log_det
        return -log_prob.mean()
