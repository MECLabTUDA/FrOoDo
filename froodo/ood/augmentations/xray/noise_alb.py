import torch
import albumentations as A
from ...augmentations import (
    OODAugmentation,
    SampableAugmentation,
)
from ...severity import ParameterSeverityMeasurement, SeverityMeasurement
from ....data import Sample
from ..utils import full_image_ood


class NoiseAAugmentation(OODAugmentation, SampableAugmentation):
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
            self.sample_intervals = [(0.001, 0.01)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class = (
            ParameterSeverityMeasurement(
                "noise_alb", (self.sample_intervals[0][0], self.sample_intervals[-1][1])
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
        # TODO: figure out how to use the sample intervals here for the var_limit
        self.transform = A.GaussNoise(mean=0, var_limit=0.001, per_channel=False, p=1)

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"noise_alb": self.sample_intervals}
        )

    def _get_parameter_dict(self):
        return {"noise_alb": self._noise}

    def _augment(self, sample: Sample) -> Sample:
        X = sample['image']
        X = X.numpy().transpose((1, 2, 0))
        img = self.transform(image=X)
        sample['image'] = torch.from_numpy(img['image'].transpose(2, 0, 1))
        sample = full_image_ood(sample, self.keep_ignored)
        return sample
