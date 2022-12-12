from .vae import VariationalAutoEncoder
# from .some_other_autoencoder import SomeOtherAutoencoder


autoencoder_name_to_class = {
    'vae': VariationalAutoEncoder,
    # 'some_other_autoencoder_name': some_other_autoencoder_class,
}


def get_autoencoder_class(autoencoder_name):
    return autoencoder_name_to_class[autoencoder_name]


def get_autoencoder_instace(encoder_config):
    autoencoder_class = get_autoencoder_class(encoder_config.encoder_name)
    return autoencoder_class(encoder_config)