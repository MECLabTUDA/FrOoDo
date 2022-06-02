import torch

from ...data.metadata import init_sample_metadata
from ...data.samples import Sample


def init_augmentation(sample: Sample) -> Sample:
    if sample["ood_mask"] == None:
        sample["ood_mask"] = create_pseudo_mask(sample["image"].shape)
    sample.metadata = init_sample_metadata(sample.metadata)
    return sample


def create_pseudo_mask(img_shape):
    return torch.ones((img_shape[1], img_shape[2]))
