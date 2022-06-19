from numpy import full
from torchvision.transforms import GaussianBlur
import torch

from copy import deepcopy

from ...augmentations import OODAugmentation, SampableAugmentation
from ....data import Sample
from ...severity import ParameterSeverityMeasurement, SeverityMeasurement
from ..utils import full_image_ood


class GaussianBlurAugmentation(OODAugmentation, SampableAugmentation):
    def __init__(
        self,
        kernel_size=(19, 19),
        sigma=2,
        sample_intervals=None,
        severity: SeverityMeasurement = None,
        keep_ignorred=True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        if sample_intervals == None:
            self.sample_intervals = [(1.2, 2.5)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class = (
            ParameterSeverityMeasurement(
                "sigma", (self.sample_intervals[0][0], self.sample_intervals[-1][1])
            )
            if severity == None
            else severity
        )
        self.keep_ignorred = keep_ignorred

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self.augmentation = GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"sigma": self.sample_intervals}
        )

    def _get_parameter_dict(self):
        return {"sigma": self.sigma}

    def _augment(self, sample: Sample) -> Sample:
        sample["image"] = self.augmentation(sample["image"])
        sample = full_image_ood(sample, self.keep_ignorred)
        return sample
