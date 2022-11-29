import torch

from enum import IntEnum


__all__ = [
    "MaskType",
    "make_channel_wise_mask",
    "make_checkerboard_mask"
]


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


def make_channel_wise_mask(x, invert_mask: bool = False):
    '''
        x: Tensor of shape (B, D)
    '''
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[..., :x.shape[-1] // 2] = True
    if invert_mask:
        mask = ~mask
    return mask


def make_checkerboard_mask(x, invert_mask: bool = False):
    '''
        x: Tensor of shape (B, D)
    '''
    mask = torch.zeros_like(x, dtype=torch.bool)
    even_ids = range(0, x.shape[-1], 2)
    mask[..., even_ids] = True
    if invert_mask:
        mask = ~mask
    return mask
    