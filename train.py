import os
import torch
import argparse
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader

from flows4ad.models import RealNVP
from flows4ad.losses import RealNVPLoss
from flows4ad.training import train

# plotting preset
sns.set(font_scale=1.3)
sns.set_style("darkgrid", {"axes.facecolor": ".95"})
# set fonttype
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family'] = 'serif'

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False

def parse_args():
    parser = argparse.ArgumentParser(description="Training Flow model for AD.")
    # Data params
    parser.add_argument(
        '--dataset', 
        type=str, 
        default="4_breastw", 
        help="Dataset.",
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True,
        help="Dataset dir. (do not add .npz extension)",
    )
    # Training params
    parser.add_argument(
        '--epochs', 
        default=10, 
        type=int, 
        help="How many epochs to train."
    )
    parser.add_argument(
        '--batch_size', 
        default=128, 
        type=int
    )
    parser.add_argument(
        '--num_workers', 
        default=4, 
        type=int
    )
    # Model parameters
    parser.add_argument(
        '--hidden_dim', 
        default=64, 
        type=int,
    )
    parser.add_argument(
        '--flow_layers', 
        default=16, 
        type=int,
    )
    parser.add_argument(
        '--mlp_layers', 
        default=2, 
        type=int,
    )
    parser.add_argument(
        '--use_channel_wise_splits', 
        action='store_true',
        help='Whether to use split channels by half'
    )
    parser.add_argument(
        '--use_checkerboard_splits', 
        action='store_true',
        help='Whether to use alternating channel split'
    )
    parser.add_argument(
        '--mlp_activation', 
        default='ReLU',
        type=str,
        help='Activation in RealNVP shift_scale MLP'
    )
    parser.add_argument(
        '--layer_norm', 
        action='store_true',
        help='Add layer norm to RealNVP shift_scale MLP'
    )
    # Optimizer params
    parser.add_argument(
        '--lr', 
        default=1e-3, 
        type=float, 
        help="Learning rate."
    )
    parser.add_argument(
        '--weight_decay', 
        default=0.0, 
        type=float, 
        help="Weight decay."
    )
    parser.add_argument(
        '--min_lr', 
        default=1e-5, 
        type=float, 
        help="Minimal learning rate."
    )
    # Logging
    parser.add_argument(
        '--log_wandb', 
        action='store_true'
    )
    parser.add_argument(
        '--log_frequency', 
        default=10,
        type=int,
        help="Batches between logging step."
    )
    # Misc params
    parser.add_argument(
        '--seed', 
        default=0, 
        type=int, 
        help="random seed."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None,
        help='Output directory where model checkpoints and results are stored.'
    )
    parser.add_argument(
        '--save_model', 
        action='store_true',
        help='Whether to save pruned model'
    )
    parser.add_argument(
        '--plot_histogram', 
        action='store_true',
        help='Whether to plot ID/OOD histogram'
    )


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # fix seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # init W&B logger
    if args.log_wandb:
        assert has_wandb
        wandb.init(config=args)

    # get data
    data = np.load(os.path.join(args.data_dir, f'{args.dataset}.npz'))
    X, y = data['X'], data['y']
    # get number of features
    num_features = X.shape[-1]

    # normalize data
    eps = 1e-6
    mean, std = X[y == 0].mean(axis=0), X[y == 0].std(axis=0) 
    X = (X - mean) / (std + eps)

    # build id and ood dataset
    id_dataset = TensorDataset(torch.from_numpy(X[y == 0]).to(torch.float32))
    ood_dataset = TensorDataset(torch.from_numpy(X[y == 1]).to(torch.float32))
    # build id and ood loader
    id_loader = DataLoader(
        id_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True
    )
    ood_loader = DataLoader(
        ood_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False
    )

    # TODO add different models
    model = RealNVP(
        # TODO some embedding dim?
        d_embed=num_features,
        d_hidden=args.hidden_dim,
        num_mlp_layers=args.mlp_layers,
        num_flow_layers=args.flow_layers,
        use_channel_wise_splits=args.use_channel_wise_splits,
        use_checkerboard_splits=args.use_checkerboard_splits,
        mlp_activation=args.mlp_activation,
        layer_norm=args.layer_norm
    ).to(device)
    # prior is N(0, 1)
    prior = Normal(loc=0.0, scale=1.0)
    loss_fn = RealNVPLoss(prior)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    train_losses, val_losses = train(
        model,
        id_loader,
        ood_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        log_frequency=args.log_frequency,
        log_wandb=args.log_wandb
    )

    # save model
    if args.save_model:
        assert args.output_dir is not None
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))
    # plot id/ood histogram
    if args.plot_histogram:
        assert args.output_dir is not None
        id_log_probs = []
        with torch.no_grad():
            for (x,) in id_loader:
                z, log_det = model(x.to(device), reverse=True)
                log_prob = prior.log_prob(z) + log_det
                id_log_probs.append(log_prob.sum(dim=-1).cpu())
        id_log_probs = torch.cat(id_log_probs, dim=0).numpy()

        ood_log_probs = []
        with torch.no_grad():
            for (x,) in ood_loader:
                z, log_det = model(x.to(device), reverse=True)
                log_prob = prior.log_prob(z) + log_det
                ood_log_probs.append(log_prob.sum(dim=-1).cpu())
        ood_log_probs = torch.cat(ood_log_probs, dim=0).numpy()

        fig, ax = plt.subplots(figsize=(9, 6))
        left = np.quantile(id_log_probs, 0.01)
        right = np.quantile(id_log_probs, 0.99)

        pretty_dataset_name = f"{args.dataset.split('_')[-1].capitalize()}"

        ax.hist(id_log_probs, bins=30, density=True, range=(left, right), alpha=0.5, color='cyan', label='ID');
        ax.hist(ood_log_probs, bins=20, density=True, range=(left, right), alpha=0.5, color='red', label='OOD');

        ax.set_xlabel(r'$\log p(x)$', fontsize=20);
        ax.legend(fontsize=20);
        ax.set_title(pretty_dataset_name, fontsize=24);

        fig.savefig(os.path.join(args.output_dir, f"{pretty_dataset_name}_id_ood_hist.pdf"))
