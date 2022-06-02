from torchvision.transforms import GaussianBlur

from frodo.data.samples.sample import Sample

from ...augmentations import OODAugmantation
from ....data import Sample
from ...severity import ParameterSeverityMeasurement, SeverityMeasurement


class GaussianBlurAugmentation(OODAugmantation):
    def __init__(self, kernel_size=(19, 19), sigma=(1.3, 2.5)) -> None:
        super().__init__()
        self.augmentation = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.severity_class = ParameterSeverityMeasurement

    def param_range(self):
        return {"sigma": (1.3, 2.5)}

    def _set_severity(self, sample: Sample) -> SeverityMeasurement:
        return super()._set_severity(sample)

    def _augment(self, sample: Sample) -> Sample:
        return super().do_random(sample, augmentation=self.augmentation)
