import random
from os import listdir
from os.path import join
from copy import deepcopy

from ...augmentations import OODAugmantation, SampledOODAugmentation
from .artifacts import ArtifactAugmentation, data_folder
from ....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from ...datatypes import DistributionSampleType


class SquamousAugmentation(OODAugmantation, ArtifactAugmentation):
    def __init__(
        self,
        scale=1,
        path=None,
        severity: SeverityMeasurement = None,
        mask_threshold=0.3,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.path = path
        self.mask_threshold = mask_threshold
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )

    def param_range(self):
        return {"scale": (0.1, 4)}

    def squamous_preproces(self, img, mask):
        return super().transparentOverlay(
            img,
            mask,
            scale=self.scale,
            overlay_path=join(
                data_folder,
                f"squamous/{random.choice(listdir(join(data_folder,'squamous')))}",
            )
            if self.path == None
            else self.path,
            mask_threshold=self.mask_threshold,
            width_slack=(-0.1, -0.1),
            height_slack=(-0.1, -0.1),
        )

    def __call__(self, img, mask):
        img, mask = self.squamous_preproces(img, mask)

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
