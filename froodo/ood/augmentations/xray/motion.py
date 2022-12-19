import torch
import torchio as tio
from ...augmentations import (
    OODAugmentation,
    SampableAugmentation,
)
from ...severity import ParameterSeverityMeasurement, SeverityMeasurement
from ....data import Sample
from ..utils import full_image_ood


class MotionAugmentation(OODAugmentation, SampableAugmentation):
    def __init__(
            self,
            motion=2,
            sample_intervals=None,
            severity: SeverityMeasurement = None,
            keep_ignored=True,
    ) -> None:
        super().__init__()
        self.motion = motion
        if sample_intervals is None:
            self.sample_intervals = [(0.2, 0.8), (1.2, 2)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class = (
            ParameterSeverityMeasurement(
                "motion",
                (self.sample_intervals[0][0], self.sample_intervals[-1][1]),
            )
            if severity is None
            else severity
        )
        self.keep_ignored = keep_ignored

    @property
    def motion(self):
        return self._motion

    @motion.setter
    def motion(self, value):
        if type(value) == tuple:
            self._motion = value
        elif type(value) == float or type(value) == int:
            assert value > 0
            self._motion = (value, value)
        self.transform = tio.transforms.RandomMotion()

    def _get_parameter_dict(self):
        return {"motion": self.motion[0]}

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"motion": self.sample_intervals}
        )

    def _augment(self, sample: Sample) -> Sample:
        x = sample["image"]
        x = torch.unsqueeze(x, dim=0)
        x = self.transform(x)
        sample["image"] = x.squeeze()

        sample = full_image_ood(sample, self.keep_ignored)
        return sample
