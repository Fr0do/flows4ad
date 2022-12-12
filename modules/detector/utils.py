from .model import FlowDetector


def get_detector_model(config):
    return FlowDetector(config.detector_config)