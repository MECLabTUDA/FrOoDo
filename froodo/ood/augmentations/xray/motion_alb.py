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
                "motionBlur",
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
        self.transform = A.MotionBlur(blur_limit=(3,7))

    def _get_parameter_dict(self):
        return {"motionBlur": self.motion[0]}

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"motionBlur": self.sample_intervals}
        )

    def _augment(self, sample: Sample) -> Sample:
        X = sample['image']
        X = X.numpy().transpose((1, 2, 0))
        img = self.transform(image=X)
        sample['image'] = torch.from_numpy(img['image'].transpose(2, 0, 1))
        sample = full_image_ood(sample, self.keep_ignored)
        return sample
