from typing import List

from ..types import Augmantation, INAugmantation, OODAugmantation, SizeAugmentation
from ....data.samples import Sample


class AugmentationPipeline(Augmantation):
    def __init__(self, augmentations) -> None:
        self.augmentations = augmentations

    def __call__(self, sample: Sample) -> Sample:
        for a in self.augmentations:
            sample = a(sample)
        return sample


class SizeInOODPipeline(AugmentationPipeline):
    def __init__(
        self,
        size_augmentations: List[SizeAugmentation] = None,
        in_augmentations: List[INAugmantation] = None,
        ood_augmentations: List[OODAugmantation] = None,
    ) -> None:
        super().__init__(
            (size_augmentations if size_augmentations != None else [])
            + (in_augmentations if in_augmentations != None else [])
            + (ood_augmentations if ood_augmentations != None else [])
        )
