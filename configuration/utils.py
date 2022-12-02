import os
import argparse

import torch
import numpy as np

import wandb


def set_experiment_logger(config=None):
    wandb.init(entity='flows4ad', project='flows4ad', config=config)


def get_device(config):
    gpu_index = config.gpu_index
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    return device


def set_random_state(config):
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_experiment_config():
    parser = argparse.ArgumentParser(description="Training Flow model for AD.")
    # Data params
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default="4_breastw", 
        help="Dataset.",
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True,
        help="Dataset dir. (do not add .npz extension)",
    )
    # Training params
    parser.add_argument(
        '--num_epochs', 
        default=10, 
        type=int, 
        help="How many epochs to train."
    )
    parser.add_argument(
        '--batch_size', 
        default=128, 
        type=int
    )
    parser.add_argument(
        '--num_workers', 
        default=4, 
        type=int
    )
    # Model parameters
    parser.add_argument(
        '--model', 
        default='real_nvp', 
        choices=['real_nvp', 'glow'],
        type=str,
    )
    parser.add_argument(
        '--hidden_dim', 
        default=64, 
        type=int,
    )
    parser.add_argument(
        '--num_flow_layers', 
        default=16, 
        type=int,
    )
    parser.add_argument(
        '--num_mlp_layers', 
        default=2, 
        type=int,
    )
    parser.add_argument(
        '--use_channel_wise_splits', 
        action='store_true',
        help='Whether to use split channels by half'
    )
    parser.add_argument(
        '--use_checkerboard_splits', 
        action='store_true',
        help='Whether to use alternating channel split'
    )
    parser.add_argument(
        '--mlp_activation', 
        default='ReLU',
        type=str,
        help='Activation in RealNVP shift_scale MLP'
    )
    parser.add_argument(
        '--layer_norm', 
        action='store_true',
        help='Add layer norm to RealNVP shift_scale MLP'
    )
    parser.add_argument(
        '--batch_norm', 
        action='store_true',
        help='Add batch norm to Glow'
    )
    # Optimizer params
    parser.add_argument(
        '--lr', 
        default=1e-3, 
        type=float, 
        help="Learning rate."
    )
    parser.add_argument(
        '--weight_decay', 
        default=0.0, 
        type=float, 
        help="Weight decay."
    )
    parser.add_argument(
        '--min_lr', 
        default=1e-5, 
        type=float, 
        help="Minimal learning rate."
    )
    # Logging
    parser.add_argument(
        '--log_wandb', 
        action='store_true'
    )
    parser.add_argument(
        '--log_frequency', 
        default=10,
        type=int,
        help="Batches between logging step."
    )
    # Misc params
    parser.add_argument(
        '--seed', 
        default=0, 
        type=int, 
        help="random seed."
    )
    parser.add_argument(
        '--gpu_index', 
        default=0, 
        type=int, 
        help="GPU index on cluster."
    )    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None,
        help='Output directory where model checkpoints and results are stored.'
    )
    parser.add_argument(
        '--save_model', 
        action='store_true',
        help='Whether to save the flow model.'
    )
    parser.add_argument(
        '--plot_histogram', 
        action='store_true',
        help='Whether to plot ID/OOD histogram.'
    )

    args = parser.parse_args()
    return args


def update_experiment_config_using_dataset(config, dataset):
    features, targets = dataset
    config.num_features = features.shape[-1]
    # maybe something else
    return config


def update_experiment_config_using_model(config, model):
    # maybe something else
    return config


def create_output_dir(config):
    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)