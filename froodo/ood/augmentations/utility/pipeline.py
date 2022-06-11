from typing import List

from ..indistribution.crop import InCrop
from ..indistribution.resize import InResize
from ..types import (
    Augmentation,
    INAugmentation,
    OODAugmentation,
    SizeAugmentation,
    AugmentationComposite,
)
from ....data.samples import Sample


class AugmentationPipeline(AugmentationComposite):
    def __init__(self, Augmentations) -> None:
        super().__init__(Augmentations)

    def __call__(self, sample: Sample) -> Sample:
        for a in self.augmentations:
            sample = a(sample)
        return sample


class SizeInOODPipeline(AugmentationPipeline):
    def __init__(
        self,
        size_augmentations: List[SizeAugmentation] = None,
        in_augmentations: List[INAugmentation] = [
            InCrop((600, 600)),
            InResize((300, 300)),
        ],
        ood_augmentations: List[OODAugmentation] = None,
    ) -> None:
        super().__init__(
            (size_augmentations if size_augmentations != None else [])
            + (in_augmentations if in_augmentations != None else [])
            + (ood_augmentations if ood_augmentations != None else [])
        )
