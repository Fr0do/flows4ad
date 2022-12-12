import os
import argparse
import yaml

import torch
import numpy as np

import wandb

from flows4ad.modules.detector import *
from flows4ad.modules.encoder import *

from flows4ad.modules.basic import *
from flows4ad.modules.embedding import *
from flows4ad.modules.loss import *


class RecursiveNamespace():
    @staticmethod
    def convert_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)
        return entry

    def __init__(self, d):
        for key, value in d.items():
            if type(value) == dict:
                setattr(self, key, RecursiveNamespace(value))
            elif type(value) == list:
                setattr(self, key, list(map(self.convert_entry, value)))
            else:
                setattr(self, key, value)


def convert_namespace_to_dict(namespace):
    return {
        key: convert_namespace_to_dict(value) 
        if type(value) == RecursiveNamespace else value
        for key, value in namespace.__dict__.items()
    }


def read_config(path):
    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def save_config(config, path):
    with open(path, 'w', encoding='utf8') as outfile:
        yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)


def get_experiment_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run_config_path', 
        type=str,
        help='Configuration file for experiment.'
    )

    options = parser.parse_args()
    config = RecursiveNamespace(read_config(options.run_config_path))
    return config


def set_experiment_logger(config):
    if config.procedure_config.log_wandb:
        wandb.init(
            entity='flows4ad', 
            project='flows4ad', 
            name=config.procedure_config.experiment_name, 
            config=config
        )


def get_device(config):
    if not hasattr(config.procedure_config, 'gpu_index'):
        config.procedure_config.gpu_index = 0
    gpu_index = config.procedure_config.gpu_index
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    return device


def set_random_state(config):
    random_seed = config.procedure_config.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def sync_experiment_config(config):
    if hasattr(config, 'detector_config'):
        experiment_name = '-'.join([
            f'dataset:{config.dataset_config.dataset_name}',
            f'flow:{config.detector_config.flow_config.flow_name}',
            f'embedding:{config.detector_config.embedding_config.embedding_name}',
            f'hidden:{config.detector_config.flow_config.hidden_dim}',
        ])
    elif hasattr(config, 'encoder_config'):
        experiment_name = '-'.join([
            f'dataset:{config.dataset_config.dataset_name}',
            f'encoder:{config.encoder_config.encoder_type}',
            f'model:{config.encoder_config.encoder_name}',
            f'hidden:{config.encoder_config.hidden_dim}',
        ])
    else:
        raise RuntimeError("No detector or encoder config. What is the expected outcome?")
    
    config.procedure_config.experiment_name = experiment_name
    config.procedure_config.output_dir = f'{config.procedure_config.output_dir}/{experiment_name}'
    return config


def update_experiment_config_using_dataset(config, dataset):
    features, targets = dataset
    if hasattr(config, 'detector_config'):
        config.detector_config.embedding_config.num_features = features.shape[-1]
        embedding_class = get_embedding_class(config.detector_config.embedding_config.embedding_name)
        config.detector_config.flow_config.num_features = embedding_class.get_embedding_size(
            config.detector_config.embedding_config
        )
    elif hasattr(config, 'encoder_config'):
        config.encoder_config.num_features = features.shape[-1]
    else:
        raise RuntimeError("No detector or encoder config. What is the expected outcome?")
    # maybe something else
    return config


def update_experiment_config_using_model(config, model):
    # maybe something else
    return config

