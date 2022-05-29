import numpy as np

import random

from ..types import OODAugmantation
from ....data.datatypes import DistributionSampleType
from ..utils import init_augmentation
from ....data.samples import Sample


class SampledOODAugmentation(OODAugmantation):
    def __init__(self, augmentation: OODAugmantation, probability=0.7) -> None:
        self.augmentation = augmentation
        self.probability = probability

        param_range_fn = getattr(self.augmentation, "param_range", None)
        assert callable(
            param_range_fn
        ), "Augmentation needs 'param_range_fn' method if you want to sample it"
        self.param_range = param_range_fn()

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

        self.sample = {}
        for key, (min_value, max_value) in self.param_range.items():
            s = np.random.uniform(min_value, max_value)
            self.sample[key] = s
            setattr(self.augmentation, key, s)
