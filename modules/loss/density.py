import torch
import torch.nn as nn


__all__ = [
    "NFLoss"
]


class NFLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.
    Args:
        prior 
        z_clamp (float): clamp z by magnitude for stability
        log_det_clamp (float): clamp log_det by magnitude for stability
    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, prior, z_clamp=None, log_det_clamp=None, nan_to_num: bool = False):
        super(NFLoss, self).__init__()
        self.prior = prior
        self.z_clamp = z_clamp
        self.log_det_clamp = log_det_clamp
        self.nan_to_num = nan_to_num

    def forward(self, z, log_det):
        if self.z_clamp is not None:
            z = torch.clamp(z, -self.z_clamp, self.z_clamp)
        if self.log_det_clamp is not None:
            log_det = torch.clamp(z, -self.log_det_clamp, self.log_det_clamp)
        if self.nan_to_num:
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            log_det = torch.nan_to_num(log_det, nan=0.0, posinf=0.0, neginf=0.0)
        log_prob = self.prior.log_prob(z) + log_det
        return -log_prob.mean()
