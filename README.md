## Flows4Ad

Repository for Anomaly Detection with Normalizing Flows.

### Data preparation

Clone ADBench repository (somewhere):
```bash
git clone https://github.com/Minqi824/ADBench
```

Data is contained in `datasets` directory. Make a symbolic link to the root 
of the current project:
```bash
ln -s <ADBench_dir/datasets> datasets
```

Datasets are expected to be in `.npz` format with `X` and `y` keys.

### Environment configuration

In order to run experiments verify 
the existence of working installation of `torch`, `joblib`, `matplotlib`, `seaborn`, `scikit-learn`.

The versions used in experiments:

torch                1.12.1
joblib               1.1.1
matplotlib           3.6.2
seaborn              0.12.1
scikit-learn         1.1.3

For logging with W&B install [wandb](https://wandb.ai/site):

```bash
pip install wandb
```

For running hyperparameter search with [Optuna](https://optuna.readthedocs.io/en/stable/) install optuna

```bash
pip install optuna
```

### Experiment running 

In order to train VAE launch `run_train_encoder.py` script with the 
specific `.yaml` config (look at the example in `configs/encoder/_reference_config.yaml`).

In order to train flow for AD detection launch `run_train_encoder.py` script with the
specific `.yaml` config (look at the example in `configs/detector/_reference_config.yaml`).
In order to change parameters of experiment edit the config file.

In order to run training with hyperparam search launch `run_train_detector_optuna.py`.

For RealNVP it is necessary to choose either `channel_wise` or `checkerboard` split or both.
