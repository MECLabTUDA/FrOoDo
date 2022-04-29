import torch

from ..metadata import init_sample_metadata


def init_augmentation(img, mask, metadata):
    if mask == None:
        mask = create_pseudo_mask(img.shape)
    return mask, init_sample_metadata(metadata)


def create_pseudo_mask(img_shape):
    return torch.ones((img_shape[1], img_shape[2]))
