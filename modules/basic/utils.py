import torch.nn as nn


def get_activation_class(activation_name: str):
    for name in dir(nn):
        if name.lower() == activation_name.lower():
            return getattr(nn, name)
    raise ValueError("Unknown activation.")
