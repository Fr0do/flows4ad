from .model import TrainableEncoder


def get_encoder_model(config):
    return TrainableEncoder(config.encoder_config)