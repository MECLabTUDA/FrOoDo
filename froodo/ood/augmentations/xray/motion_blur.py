import torch
import albumentations as A
from ...augmentations import (
    OODAugmentation,
    SampableAugmentation,
)
from ...severity import ParameterSeverityMeasurement, SeverityMeasurement
from ....data import Sample
from ..utils import full_image_ood


class MotionBlurAugmentation(OODAugmentation, SampableAugmentation):
    def __init__(
        self,
        motion=5,
        sample_intervals=None,
        severity: SeverityMeasurement = None,
        keep_ignored=True,
    ) -> None:
        super().__init__()
        self.motion = motion
        if sample_intervals is None:
            self.sample_intervals = [(5, 20)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class = (
            ParameterSeverityMeasurement(
                "motion",
                (0, 1),  # uses a normalized range
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
        value = int(value)
        if value % 2 == 0:
            value = value + 1
        assert value > 3
        self._motion = value
        self.transform = A.MotionBlur(
            blur_limit=(value, value), allow_shifted=True, p=1
        )

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"motion": self.sample_intervals}
        )

    def _get_parameter_dict(self):
        return {"motion": self.normalized_param_range(self.motion)}

    def _augment(self, sample: Sample) -> Sample:
        X = sample["image"]
        X = X.numpy().transpose((1, 2, 0))
        img = self.transform(image=X)
        sample["image"] = torch.from_numpy(img["image"].transpose(2, 0, 1))
        sample = full_image_ood(sample, self.keep_ignored)
        return sample

    def normalized_param_range(self, value: int):
        return (value - self.sample_intervals[0][0]) / (
            self.sample_intervals[-1][1] - self.sample_intervals[0][0]
        )
