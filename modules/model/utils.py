from .model import GeneralModel


def get_model(config):
    return GeneralModel(config.model_config)