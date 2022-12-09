import torch.nn as nn

class VAELoss(nn.Module):

    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")
    
    # x_recon ist der im forward im Model erstellte recon_batch, x ist der originale x Batch, mu ist mu und logvar ist logvar 
    def forward(self, x_recon, x, mu, log_sigma):
        mse_loss = self.mse_loss(x_recon, x)
        kl_loss = -0.5 * (1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp()).sum(dim=-1).mean()
        return mse_loss + kl_loss 
        