from ..types import Augmentation, SampableAugmentation, OODAugmentation
from ....data.datatypes import DistributionSampleType
from ....data.samples import Sample


class PartialOODAugmentaion(OODAugmentation):
    def __init__(self, base_augmentation, mode="linear") -> None:
        super().__init__()
        self.base_augmentation = base_augmentation
        self.mode = mode

    def _augment(self, sample: Sample) -> Sample:
        return super()._augment(sample)
