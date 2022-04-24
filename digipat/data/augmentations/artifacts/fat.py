from os.path import join
from copy import deepcopy

from ...augmentations import OODAugmantation, SampledOODAugmentation
from .artifacts import ArtifactAugmentation, data_folder
from ...severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from ...datatypes import DistributionSampleType


class FatAugmentation(OODAugmantation, ArtifactAugmentation):
    def __init__(
        self, scale=1, severity: SeverityMeasurement = None, mask_threshold=0.4
    ) -> None:
        super().__init__()
        self.scale = scale
        self.mask_threshold = mask_threshold
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )

    def __call__(self, img, mask):
        img, mask = super().transparentOverlay(
            img,
            mask,
            overlay_path=join(data_folder, "fat/fat.png"),
            scale=self.scale,
            mask_threshold=self.mask_threshold,
            width_slack=(-0.1, -0.1),
            height_slack=(-0.1, -0.1),
        )

        severity: SeverityMeasurement = deepcopy(self.severity_class)
        severity.calculate_measurement(img, mask, {"scale": self.scale})

        return (
            img,
            mask,
            {
                "type": DistributionSampleType.IN_DISTRIBUTION_DATA
                if severity.get_bin(ignore_true_bin=True) == -1
                else DistributionSampleType.AUGMENTATION_OOD_DATA,
                "severity": severity,
            },
        )
