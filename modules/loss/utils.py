from .density import NFLoss
from .vae import VAELoss


def get_detector_loss(prior, config=None):
    loss = NFLoss(
        prior, 
        getattr(config.optimisation_config, 'z_clamp', None),
        getattr(config.optimisation_config, 'log_det_clamp', None),
        getattr(config.optimisation_config, 'nan_to_num', False)
    )
    return loss


encoder_name_to_loss_class = {
    'vae': VAELoss
}


def get_encoder_loss(config):
    loss_class = encoder_name_to_loss_class[config.encoder_config.encoder_name]
    loss_instance = loss_class()
    return loss_instance
