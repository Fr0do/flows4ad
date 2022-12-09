import os

import torch
import torch.optim as optim

from execution import engine, engine_vae
from modules import *
from modules.vae import *
from visualisation import (
    visualise_prediction_histograms, 
    set_visualisation_options, 
    save_performance_metrics,
)

def get_optimizer(model, config):
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.optimisation_config.lr, 
        weight_decay=config.optimisation_config.weight_decay
    )
    return optimizer


def get_scheduler(optimizer, config):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.optimisation_config.num_steps, 
        eta_min=config.optimisation_config.min_lr
    )
    return scheduler


def save_model(model, config, checkpoint_name='model.pth'):
    torch.save(model.state_dict(), os.path.join(config.procedure_config.output_dir, checkpoint_name))


def collect_predictions(model, prior, dataloader, device):
    log_probs = []
    with torch.no_grad():
        for (x,) in dataloader:
            z, log_det = model(x.to(device), reverse=True)
            log_prob = prior.log_prob(z) + log_det
            log_probs.append(log_prob.sum(dim=-1).cpu())
    
    log_probs = torch.cat(log_probs, dim=0).numpy()
    return log_probs


def setup_before_experiment(environment, config):
    model, prior, loss, device, dataloaders, optimizer, scheduler = environment
    # maybe something else
    
    output_dir =  getattr(config.procedure_config, 'output_dir')
    if output_dir:
        os.makedirs(config.procedure_config.output_dir, exist_ok=True)

    save_latents_dir = getattr(config.procedure_config, 'save_latents_dir', None)
    if save_latents_dir:
        assert getattr(config, 'vae_config', None) is not None
        os.makedirs(save_latents_dir, exist_ok=True)
        

def run_experiment(environment, config):
    model, prior, loss, device, dataloaders, optimizer, scheduler = environment

    results = engine.train(
        model, dataloaders['in'], dataloaders['out'],
        steps=config.optimisation_config.num_steps,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss,
        device=device,
        log_frequency=config.procedure_config.log_frequency,
        log_wandb=config.procedure_config.log_wandb
    )

    return results


# TODO naming is awful, I undersantd, I do not have better idea how to name it
def run_vae_training(environment, config):
    model, prior, loss, device, dataloaders, optimizer, scheduler = environment

    results = engine_vae.train_vae(
        model, dataloaders['in'], dataloaders['out'],
        steps=config.optimisation_config.num_steps,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss,
        device=device,
        log_frequency=config.procedure_config.log_frequency,
        log_wandb=config.procedure_config.log_wandb
    )

    if hasattr(config.procedure_config, 'save_latents_dir'):
        engine_vae.save_latents(
            model,
            dataloaders['in'], 
            dataloaders['out'],
            os.path.join(
                config.procedure_config.save_latents_dir, 
                f'{config.dataset_config.dataset_name}.npz',
            ),
            device=device
        )

    return results


def teardown_after_experiment(results, environment, config):
    model, prior, loss, device, dataloaders, optimizer, scheduler = environment
    # maybe something else
    
    # save model
    if config.procedure_config.save_model:
        assert config.procedure_config.output_dir is not None
        save_model(model, config)
    
    # plot id/ood histogram
    if config.procedure_config.plot_histograms:
        assert config.procedure_config.output_dir is not None

        predictions = {}
        for domain_name in ['in', 'out']:
            dataloader = dataloaders[domain_name]
            log_probs = collect_predictions(model, prior, dataloader, device)
            predictions[domain_name] = log_probs

        set_visualisation_options()
        visualise_prediction_histograms(predictions, config)

    if config.procedure_config.save_metrics:
        assert config.procedure_config.output_dir is not None
        save_performance_metrics(results, config, keys=['AUROC'])
        