import wandb
import torch
import numpy as np

from tqdm import tqdm
from .utils import make_stepwise_generator, make_drop_last_false_loader


__all__ = ["train_step", "train_epoch", "eval_epoch", "train"]


def train_step(model, x, loss_fn, optimizer, device='cpu'):
    model.train()
    x = x.to(device)
    x_recon, mu, log_sigma = model(x)
    loss = loss_fn(x_recon, x, mu, log_sigma)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device='cpu'):
    model.eval()
    stats = []
    for i, (x,) in enumerate(loader):
        x = x.to(device)
        x_recon, mu, log_sigma = model(x)
        loss = loss_fn(x_recon, x, mu, log_sigma)
        stats.append(loss.item())
    return np.mean(stats)


def train(
    model,
    train_loader,
    test_loader,
    steps: int,
    optimizer,
    loss_fn, 
    scheduler = None,
    device: str = 'cpu',
    log_wandb: bool = False,
    log_frequency: int = 50,
    tqdm_bar: bool = False
):
    train_losses, test_losses = [], []
    progress_bar = tqdm(range(1, steps + 1)) if tqdm_bar else range(1, steps + 1)
    # make step-wise loader
    train_generator = make_stepwise_generator(train_loader, steps)
    # compute running train loss
    running_train_losses = []

    for step in progress_bar:
        # make train step
        (x,) = next(iter(train_generator))
        train_loss = train_step(model, x, loss_fn, optimizer, device)
        # update loss history
        train_losses += [train_loss]
        running_train_losses += [train_loss]
        if step % log_frequency == 0:
            test_loss = eval_epoch(model, test_loader, loss_fn, device)
            test_losses += test_loss
            train_loss_avg = np.mean(running_train_losses)
            print('-' * 10)
            print(f'Step [{step:>2}/{steps:>2}] | Train loss: {train_loss_avg:.3f} Test loss: {test_loss:.3f}')
            print('-' * 10)
            # log to wandb
            if log_wandb:
                wandb.log({'train_loss': train_loss_avg, 'test_loss': test_loss}, step=step)
        # make scheduler step (optional)
        if scheduler:
            scheduler.step()
        
    # compute final metrics
    return {
        'train_loss': train_losses, 
        'test_loss': test_losses
    }
    

@torch.no_grad()
def save_latents(
    model, 
    train_loader, 
    test_loader,
    latents_path: str,
    device: str = 'cpu'
):  
    labels = np.concatenate(
        [np.zeros(len(train_loader.dataset)), np.ones(len(test_loader.dataset))],
        axis=0
    )

    train_loader_copy = make_drop_last_false_loader(train_loader)

    latents = []
    for loader in [train_loader_copy, test_loader]:
        for (x,) in loader:
            x = x.to(device)
            z, _ = model.encode(x)
            latents.append(z.cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    np.savez(latents_path, X=latents, y=labels)
