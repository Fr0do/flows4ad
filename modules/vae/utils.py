from .vae import VariationalAutoEncoder

def get_vae_model(config):
    return VariationalAutoEncoder.from_config(config.vae_config)
