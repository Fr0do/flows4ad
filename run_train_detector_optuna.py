import os
import subprocess

os.environ['WANDB_ENTITY'] = 'flows4ad'
os.environ['WANDB_PROJECT'] = 'flows4ad'

run_config_path = './configs/detector/47_yeast/features/maf_config.yaml'

subprocess.call([
    'python', 'train_detector_optuna.py', 

    '--run_config_path', run_config_path,
    '--num_trials', str(20),
])