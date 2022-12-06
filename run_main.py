import subprocess

run_config_path = './configuration/config.yaml'

subprocess.call([
    'python', 'main.py', 

    '--run_config_path', run_config_path,
])