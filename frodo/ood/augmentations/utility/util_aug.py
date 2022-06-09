import random

from ...augmentations import OODAugmentation
from ....data.samples import Sample


class ProbabilityAugmentation(OODAugmentation):
    def __init__(self, augmentation, prob=0.5) -> None:
        super().__init__()
        assert isinstance(augmentation, OODAugmentation)
        self.augmentation = augmentation
        self.prob = prob

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample
        return self.augmentation(sample)


class Nothing(OODAugmentation):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, sample: Sample) -> Sample:
        return sample


class NTimesAugmentation(OODAugmentation):
    def __init__(self, augmentation, n) -> None:
        super().__init__()
        self.augmentation = augmentation
        self.n = n

    def __call__(self, sample: Sample) -> Sample:
        for _ in range(self.n):
            sample = self.augmentation(sample)
        return sample
