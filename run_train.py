import os
import shutil
import subprocess

dataset_name = '6_cardio'
data_dir = 'datasets/Classical'

batch_size = 64
num_workers = 4

num_flow_layers = 16
num_mlp_layers = 2

lr = 1e-3
seed = 42

experiment_dir = f'results/{dataset_name}'

subprocess.call([
    'python', 'train.py', 

    '--dataset_name', dataset_name, 
    '--data_dir', data_dir, 
    
    '--batch_size', str(batch_size),
    '--num_workers', str(num_workers),
    
    '--num_flow_layers', str(num_flow_layers),
    '--num_mlp_layers', str(num_mlp_layers),
    '--layer_norm',
    
    '--use_channel_wise_splits',
    '--use_checkerboard_splits',
    
    '--lr', str(lr),
    '--seed', str(seed),
    '--output_dir', experiment_dir,

    '--plot_histogram',
    '--log_wandb',
])