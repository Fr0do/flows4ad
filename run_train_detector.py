import os
import sys
import subprocess


sys.path.append('..')

current_config_path = f'./configs/detector/11_donors/features/maf_config.yaml'

subprocess.call([
    'python', 'train_detector.py', 
    '--run_config_path', current_config_path,
])