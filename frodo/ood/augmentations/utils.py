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


def full_image_ood(sample: Sample, keep_ignorred):
    if keep_ignorred:
        ignore_map = sample["ood_mask"] == sample.metadata["ignore_index"]
    sample["ood_mask"] = torch.zeros_like(sample["ood_mask"])
    if keep_ignorred:
        sample["ood_mask"][ignore_map] = sample.metadata["ignore_index"]
    return sample
