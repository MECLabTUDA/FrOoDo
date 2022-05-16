import torch

from .metadata import SampleMetdadataBatch
from .samples import Batch


def sample_collate(d):
    return Batch.init_from_samples(d)


def augmentation_collate(d):
    img = []
    masks = []
    meta = SampleMetdadataBatch([])
    for i, m, me in d:
        img.append(i)
        masks.append(m)
        meta.batch.append(me)
    return torch.stack(img), torch.stack(masks), meta
