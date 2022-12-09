import subprocess

run_config_path = './configs/vae/example_vae_config.yaml'

subprocess.call([
    'python', 'train_vae.py', 

    '--run_config_path', run_config_path,
])