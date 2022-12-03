from data import (
    load_dataset, 
    process_dataset, 
    get_datasets, 
    get_dataloaders
)
from configuration import (
    get_experiment_config, 
    get_device, 
    set_random_state, 
    set_experiment_logger, 
    update_experiment_config_using_dataset, 
    update_experiment_config_using_model
)
from experiment import (
    get_model, 
    get_prior, 
    get_loss, 
    get_optimizer,
    get_scheduler, 
    setup_before_experiment, 
    run_experiment, 
    teardown_after_experiment
)


if __name__ == "__main__":
    config = get_experiment_config()
    device = get_device(config)

    set_random_state(config)
    if config.log_wandb:
        set_experiment_logger(config)

    dataset = load_dataset(config)
    dataset, scaler = process_dataset(dataset, config)
    config = update_experiment_config_using_dataset(config, dataset)

    datasets = get_datasets(dataset, config)
    dataloaders = get_dataloaders(datasets, config)

    model = get_model(config)
    config = update_experiment_config_using_model(config, dataset)
    model.to(device)
    
    prior = get_prior(config)
    loss = get_loss(prior, config)
    
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    environment = model, prior, loss, device, dataloaders, optimizer, scheduler
    
    setup_before_experiment(environment, config)
    results = run_experiment(environment, config)
    teardown_after_experiment(environment, config)
