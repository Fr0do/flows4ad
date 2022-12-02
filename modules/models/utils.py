import torch
import torch.distributions as dist

from . import flow


def get_model(config):
    # TODO add different models
    if config.model == 'real_nvp':
        model = flow.RealNVP(
            # TODO some embedding dim?
            d_embed=config.num_features,
            d_hidden=config.hidden_dim,

            num_mlp_layers=config.num_mlp_layers,
            num_flow_layers=config.num_flow_layers,
            
            use_channel_wise_splits=config.use_channel_wise_splits,
            use_checkerboard_splits=config.use_checkerboard_splits,
            
            mlp_activation=config.mlp_activation,
            layer_norm=config.layer_norm
        )

    elif config.model == 'glow':
        model = flow.Glow(
            # TODO some embedding dim?
            d_embed=config.num_features,
            d_hidden=config.hidden_dim,
            
            num_mlp_layers=config.num_mlp_layers,
            num_flow_layers=config.num_flow_layers,
            
            use_channel_wise_splits=config.use_channel_wise_splits,
            use_checkerboard_splits=config.use_checkerboard_splits,
            
            mlp_activation=config.mlp_activation,
            layer_norm=config.layer_norm,
            batch_norm=config.batch_norm
        )
    
    else:
        raise NotImplementedError("Unknown model")

    return model


def get_prior(config=None):
    prior = dist.Normal(loc=0.0, scale=1.0)
    return prior
