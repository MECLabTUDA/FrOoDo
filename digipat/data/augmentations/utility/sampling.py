import numpy as np

import random

from ..aug import OODAugmantation
from ...datatypes import DistributionSampleType
from ..utils import init_augmentation


class SampledOODAugmentation(OODAugmantation):
    def __init__(self, augmentation: OODAugmantation, probability=0.7) -> None:
        self.augmentation = augmentation
        self.probability = probability

        param_range_fn = getattr(self.augmentation, "param_range", None)
        assert callable(
            param_range_fn
        ), "Augmentation needs 'param_range_fn' method if you want to sample it"
        self.param_range = param_range_fn()

    def __call__(self, img, mask, metadata=None):
        mask, metadata = init_augmentation(img, mask, metadata)
        self.sample_params()
        if self.skip:
            # if no augmentation is applied, data remains IN Distribution
            metadata["type"] = DistributionSampleType.IN_DATA
            return img, mask, metadata

        i = 0
        while True:
            # if augmentation should be applied, try at most 10 times to get an OOD Sample
            # This my be necessary if the artifact is places outside the patch
            _img, _mask, _metadata = self.augmentation(img, mask, metadata)

            if _metadata["type"] is DistributionSampleType.OOD_DATA or i >= 10:
                break
            self.sample_params()
            i += 1

        return _img, _mask, _metadata

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
