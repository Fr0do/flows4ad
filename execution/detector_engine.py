import wandb
import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from .utils import make_stepwise_generator, make_drop_last_false_loader


__all__ = ["train_step", "train_epoch", "eval_epoch", "train"]


def train_step(model, x, loss_fn, optimizer, device='cpu'):
    model.train()
    x = x.to(device)
    z, log_det = model(x, reverse=True)
    loss = loss_fn(z, log_det)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(model, loader, loss_fn, optimizer, log_frequency: int, device='cpu'):
    model.train()
    stats = []
    for i, (x,) in enumerate(loader):
        x = x.to(device)
        z, log_det = model(x, reverse=True)
        loss = loss_fn(z, log_det)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i % log_frequency == 0:
            print(f'Step [{i:>2}/{len(loader):>2}] Loss {loss.item():.3f}')
        stats.append(loss.item())
    return stats


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device='cpu'):
    model.eval()
    stats = []
    for i, (x,) in enumerate(loader):
        x = x.to(device)
        z, log_det = model(x, reverse=True)
        loss = loss_fn(z, log_det)
        stats.append(loss.item())
    return np.mean(stats)


# TODO add metric computation during training?
@torch.no_grad()
def estimate_ad_performance(model, train_loader, val_loader, prior, device='cpu'):
    y_pred = []
    y_true = np.concatenate(
        [np.zeros(len(train_loader.dataset)), np.ones(len(val_loader.dataset))],
        axis=0
    )
    train_loader_copy = make_drop_last_false_loader(train_loader)
    for loader in [train_loader_copy, val_loader]:
        for i, (x,) in enumerate(loader):
            x = x.to(device)
            z, log_det = model(x, reverse=True)
            # high log_prob shoud correspond to class 0, low to 1
            log_prob = (prior.log_prob(z) + log_det).sum(dim=-1)
            y_pred.append(-log_prob.cpu().numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    return roc_auc_score(y_true, y_pred)


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
    
    auroc = estimate_ad_performance(model, train_loader, test_loader, loss_fn.prior, device=device)
    print(f'AUROC in anomaly detection: {auroc:.3f}')
    if log_wandb:
        wandb.log({'AUROC': auroc})
    
    # compute final metrics
    return {
        'train_loss': train_losses, 
        'test_loss': test_losses,
        'AUROC': auroc
    }
