import os
import sys
import subprocess


sys.path.append('..')

current_config_path = f'./configs/encoder/_reference_config.yaml'

subprocess.call([
    'python', 'train_encoder.py', 
    '--run_config_path', current_config_path,
])