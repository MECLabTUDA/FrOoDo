import torch

from .metadata import SampleMetdadataBatch


def augmentation_collate(d):
    img = []
    masks = []
    meta = SampleMetdadataBatch([])
    for i, m, me in d:
        img.append(i)
        masks.append(m)
        meta.batch.append(me)
    return torch.stack(img), torch.stack(masks), meta
    # return (
    #    torch.stack([i for i, _, _ in d]),
    #    torch.stack([m for _, m, _ in d]),
    #    [meta for _, _, meta in d],
    # )
