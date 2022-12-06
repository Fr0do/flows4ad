import wandb
import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import roc_auc_score


__all__ = ["train_epoch", "eval_epoch", "train"]


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
    return stats


# TODO add metric computation during training?
@torch.no_grad()
def estimate_ad_performance(model, train_loader, val_loader, prior, device='cpu'):
    y_pred = []
    y_true = np.concatenate(
        [np.zeros(len(train_loader.dataset)), np.ones(len(val_loader.dataset))],
        axis=0
    )
    # temporarily do not drop last
    template = dict(train_loader.__dict__)
    # drop attributes that will be auto-initialized
    to_drop = [k for k in template if k.startswith("_") or k == "batch_sampler"]
    for item in to_drop:
        template.pop(item)
    template['drop_last'] = False
    train_loader_copy = type(train_loader)(**template)
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
    epochs: int,
    optimizer,
    loss_fn, 
    scheduler = None,
    device: str = 'cpu',
    log_wandb: bool = False,
    log_frequency: int = 50,
    tqdm_bar: bool = False
):
    train_losses, test_losses = [], []
    progress_bar = tqdm(range(epochs)) if tqdm_bar else range(epochs)
    for epoch in progress_bar:
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, log_frequency, device)
        test_loss = eval_epoch(model, test_loader, loss_fn, device)
        # update loss history
        train_losses += train_loss
        test_losses += test_loss
        print('-' * 10)
        print(f'Epoch [{epoch:>2}/{epochs:>2}] | Train loss: {np.mean(train_loss):.3f} Test loss: {np.mean(test_loss):.3f}')
        print('-' * 10)
        # make scheduler step (optional)
        if scheduler:
            scheduler.step()
        # log to wandb
        if log_wandb:
            wandb.log({'train_loss': np.mean(train_loss), 'test_loss': np.mean(test_loss)}, step=epoch)
    
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
