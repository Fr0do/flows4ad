import os
import sys
import optuna
import joblib
sys.path.append('..')

from data import (
    load_dataset, 
    process_dataset, 
    get_datasets, 
    get_dataloaders
)
from configuration import (
    get_experiment_config, 
    sync_experiment_config,
    get_device, 
    set_random_state, 
    set_experiment_logger, 
    update_experiment_config_using_dataset, 
    update_experiment_config_using_model
)
from experiment import (
    get_detector_model, 
    get_detector_prior, 
    get_detector_loss, 
    get_optimizer,
    get_scheduler, 
    setup_before_experiment, 
    run_detector_experiment,
    teardown_after_experiment
)
from functools import partial


def objective(trial: optuna.trial.Trial, environment, config):
    # model params
    num_layers = trial.suggest_int("num_layers", 4, 24, step=4)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 512, step=32)
    num_mlp_layers = trial.suggest_int("num_mlp_layers", 2, 3, step=1)
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "tanh"])
    # optimisation params
    num_steps = trial.suggest_int("num_steps", 100, 800, step=50)
    batch_size = trial.suggest_int("batch_size", 8, 256, log=True)
    lr = trial.suggest_float("lr", 1e-5, 1e-3)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    # edit model config
    config.detector_config.flow_config.num_layers = num_layers
    config.detector_config.flow_config.hidden_dim = hidden_dim
    config.detector_config.flow_config.num_mlp_layers = num_mlp_layers
    config.detector_config.flow_config.activation = activation
    # edit optimiazation config
    config.optimisation_config.batch_size = batch_size
    config.optimisation_config.num_steps = num_steps
    config.optimisation_config.lr = lr
    config.optimisation_config.weight_decay = weight_decay
    # create dataloaders
    dataloaders = get_dataloaders(datasets, config)
     # create model
    model = get_detector_model(config)
    config = update_experiment_config_using_model(config, model)
    model.to(device)
    # create optimizer
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    # edit environment
    environment = model, prior, loss, device, dataloaders, optimizer, scheduler
    results = run_detector_experiment(environment, config)
    return results['AUROC']


if __name__ == "__main__":
    config = get_experiment_config()
    # sanity check
    assert config.num_trials is not None, \
        "One has to specify number of trials for Optuna search"
    config = sync_experiment_config(config)
    device = get_device(config)

    set_random_state(config)
    set_experiment_logger(config)

    dataset = load_dataset(config)
    dataset, scaler = process_dataset(dataset, config)
    config = update_experiment_config_using_dataset(config, dataset)

    datasets = get_datasets(dataset, config)
    
    prior = get_detector_prior(config)
    loss = get_detector_loss(prior, config)
    
    environment = None, prior, loss, device, None, None, None
    
    setup_before_experiment(environment, config)
    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, config=config, environment=environment), n_trials=config.num_trials)
    joblib.dump(study, os.path.join(config.procedure_config.output_dir, "optuna_study.pkl"))
    # add something ?
