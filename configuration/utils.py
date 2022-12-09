import os
import argparse
from attr import has
import yaml

import torch
import numpy as np

import wandb

from flows4ad.modules.basic import *
from flows4ad.modules.embedding import *
from flows4ad.modules.flow import *
from flows4ad.modules.model import *
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


def read_config(path):
    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


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
    if hasattr(config, 'model_config'):
        experiment_name = '-'.join([
            f'dataset:{config.dataset_config.dataset_name}',
            f'flow:{config.model_config.flow_config.flow_name}',
            f'embedding:{config.model_config.embedding_config.embedding_name}',
        ])
    else:
        experiment_name = '-'.join([
            f'dataset:{config.dataset_config.dataset_name}',
            f'VAE training'
        ])
    config.procedure_config.experiment_name = experiment_name
    config.procedure_config.output_dir = f'results/{experiment_name}'
    return config


def update_experiment_config_using_dataset(config, dataset):
    features, targets = dataset
    if hasattr(config, 'model_config'):
        config.model_config.embedding_config.num_features = features.shape[-1]
        embedding_class = get_embedding_class(config.model_config.embedding_config.embedding_name)
        config.model_config.flow_config.num_features = embedding_class.get_embedding_size(
            config.model_config.embedding_config
        )
    elif hasattr(config, 'vae_config'):
        config.vae_config.d_in = features.shape[-1]
    else:
        raise RuntimeError("No flow model and vae config. What is the expected outcome? ")
    # maybe something else
    return config


def update_experiment_config_using_model(config, model):
    # maybe something else
    return config


def create_output_dir(config):
    if config.procedure_config.output_dir:
        os.makedirs(config.procedure_config.output_dir, exist_ok=True)
