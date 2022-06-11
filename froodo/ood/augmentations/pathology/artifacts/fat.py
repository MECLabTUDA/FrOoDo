from os.path import join
from copy import deepcopy

from ....augmentations import OODAugmentation, SampableAugmentation
from .artifacts import ArtifactAugmentation, data_folder
from .....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from .....data.datatypes import DistributionSampleType
from .....data.samples import Sample


class FatAugmentation(OODAugmentation, ArtifactAugmentation, SampableAugmentation):
    def __init__(
        self,
        scale=1,
        severity: SeverityMeasurement = None,
        mask_threshold=0.4,
        sample_intervals=None,
        keep_ignorred=True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.mask_threshold = mask_threshold
        if sample_intervals == None:
            self.sample_intervals = [(0.1, 5)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        self.keep_ignorred = keep_ignorred

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"scale": self.sample_intervals}
        )

    def _augment(self, sample: Sample) -> Sample:
        img, mask = super().transparentOverlay(
            sample["image"],
            sample["ood_mask"],
            scale=self.scale,
            mask_threshold=self.mask_threshold,
            overlay_path=join(data_folder, "fat", "fat.png"),
            width_slack=(-0.1, -0.1),
            height_slack=(-0.1, -0.1),
            ignore_index=None
            if not self.keep_ignorred
            else sample.metadata["ignore_index"],
        )
        sample["image"] = img
        sample["ood_mask"] = mask

        return sample
