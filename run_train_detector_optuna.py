import os
import subprocess

os.environ['WANDB_ENTITY'] = 'flows4ad'
os.environ['WANDB_PROJECT'] = 'flows4ad'

run_config_path = f'./configs/detector/_reference_config.yaml'

subprocess.call([
    'python', 'train_detector_optuna.py', 

    '--run_config_path', run_config_path,
    '--num_trials', str(20),
])