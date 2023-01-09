import torchio as tio
import torch
from ...augmentations import (
    OODAugmentation,
    SampableAugmentation,
)
from ...severity import ParameterSeverityMeasurement, SeverityMeasurement
from ....data import Sample
from ..utils import full_image_ood


class NoiseAugmentation(OODAugmentation, SampableAugmentation):
    def __init__(
            self,
            noise=(0, 0.25),
            sample_intervals=None,
            severity: SeverityMeasurement = None,
            keep_ignored=True,
    ) -> None:
        super().__init__()
        self.noise = noise
        if sample_intervals is None:
            self.sample_intervals = [(1.2, 2.5)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class = (
            ParameterSeverityMeasurement(
                "noise", (self.sample_intervals[0][0], self.sample_intervals[-1][1])
            )
            if severity is None
            else severity
        )
        self.keep_ignored = keep_ignored

    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, value):
        if type(value) == tuple:
            self._noise = value
        elif type(value) == float or type(value) == int:
            assert value > 0
            self._noise = (value, value)
        self.transform = tio.transforms.RandomNoise(mean=self.noise[0], std=self.noise[1])

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"noise": self.sample_intervals}
        )

    def _get_parameter_dict(self):
        return {"noise": self._noise}

    def _augment(self, sample: Sample) -> Sample:
        imag = sample["image"]
        imag = torch.unsqueeze(imag, dim=-1)
        imag = self.transform(imag)
        sample["image"] = imag.squeeze()
        sample = full_image_ood(sample, self.keep_ignored)
        return sample
