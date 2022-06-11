import random
from os import listdir
from os.path import join


from ....augmentations import OODAugmentation
from .artifacts import ArtifactAugmentation, data_folder
from .....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from .....data.metadata import *
from ...utils import *
from .....data.samples import Sample


class BloodCellAugmentation(ArtifactAugmentation, OODAugmentation):
    def __init__(
        self,
        scale=1,
        severity: SeverityMeasurement = None,
        path=None,
        mask_threshold=0.5,
        sample_range=None,
        keep_ignorred=True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.path = path
        self.mask_threshold = mask_threshold
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        if sample_range == None:
            self.sample_range = {"scale": (0.5, 0.9)}
        else:
            self.sample_range = sample_range
        self.keep_ignorred = keep_ignorred

    def param_range(self):
        return self.sample_range

    def _augment(self, sample: Sample) -> Sample:
        for i in range(10):
            img, mask = super().transparentOverlay(
                sample["image"],
                sample["ood_mask"],
                scale=self.scale,
                mask_threshold=self.mask_threshold,
                overlay_path=join(
                    "D:\OOD_Data\BCSS\ood-ness\Artefacts\Blood\cell",
                    random.choice(
                        listdir("D:\OOD_Data\BCSS\ood-ness\Artefacts\Blood\cell")
                    ),
                )
                if self.path == None
                else self.path,
                width_slack=(-0.1, -0.1),
                height_slack=(-0.1, -0.1),
                ignore_index=None
                if not self.keep_ignorred
                else sample.metadata["ignore_index"],
            )
            sample["image"] = img
            sample["ood_mask"] = mask

        return sample