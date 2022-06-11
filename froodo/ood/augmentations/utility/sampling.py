import numpy as np

import random

from ..types import Augmentation, SampableAugmentation
from ....data.datatypes import DistributionSampleType
from ....data.samples import Sample
from ..utils import init_augmentation


class SampledAugmentation(Augmentation):
    def __init__(self, augmentation: SampableAugmentation, probability=0.7) -> None:
        assert isinstance(
            augmentation, SampableAugmentation
        ), "Augmentation that should be sampled needs to be instance of SampableAugmentation"
        self.augmentation: SampableAugmentation = augmentation
        self.probability = probability

    def __call__(self, sample: Sample):
        sample = init_augmentation(sample)
        self.sample_params()
        if self.skip:
            return sample

        i = 0
        while True:
            # if augmentation should be applied, try at most 10 times to get an OOD Sample
            # This my be necessary if the artifact is places outside the patch
            sample = self.augmentation(sample)

            if sample.metadata["type"] is DistributionSampleType.OOD_DATA or i >= 10:
                break
            self.sample_params()
            i += 1

        return sample

    def sample_params(self):
        if random.random() >= self.probability:
            self.skip = True
            return
        else:
            self.skip = False

        self.augmentation._apply_sampling()
