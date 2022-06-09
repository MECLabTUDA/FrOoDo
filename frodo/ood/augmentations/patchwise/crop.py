import torch
from torchvision.transforms import RandomCrop, Resize

import random

from ...augmentations import OODAugmentation, SampableAugmentation, InCrop, InResize
from ....data import Sample
from ...severity import ParameterSeverityMeasurement, SeverityMeasurement
from ..utils import full_image_ood


class ZoomInAugmentation(OODAugmentation, SampableAugmentation):
    def __init__(
        self,
        crop=0.5,
        sample_intervals=None,
        severity: SeverityMeasurement = None,
        keep_ignorred=True,
    ) -> None:
        super().__init__()
        assert crop > 0 and crop <= 1
        self.crop = crop
        if sample_intervals == None:
            self.sample_intervals = [(0.2, 0.9)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class = (
            ParameterSeverityMeasurement(
                "crop", (self.sample_intervals[0][0], self.sample_intervals[-1][1])
            )
            if severity == None
            else severity
        )
        self.keep_ignorred = keep_ignorred

    def _get_parameter_dict(self):
        return {"crop": self.crop}

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"crop": self.sample_intervals}
        )

    def _augment(self, sample: Sample) -> Sample:

        _, h, w = sample.image.shape
        sample = InCrop((round(self.crop * h), round(self.crop * w)))(sample)
        sample = InResize((h, w))(sample)

        sample = full_image_ood(sample, self.keep_ignorred)
        return sample
