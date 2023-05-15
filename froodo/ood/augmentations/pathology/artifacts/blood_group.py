import random
from os import listdir
from os.path import join
from copy import deepcopy

from ....augmentations import OODAugmentation, SampableAugmentation
from .artifacts import ArtifactAugmentation, data_folder
from .....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from .....data.datatypes import DistributionSampleType
from .....data.samples import Sample


class BloodGroupAugmentation(
    OODAugmentation, ArtifactAugmentation, SampableAugmentation
):
    def __init__(
        self,
        scale=1,
        num_groups=None,
        path=None,
        severity: SeverityMeasurement = None,
        mask_threshold=0.5,
        sample_intervals=None,
        keep_ignorred=True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.path = path
        self.mask_threshold = mask_threshold
        self.num_groups = 1
        if num_groups == None:
            self.num_groups_sample_intervals = [(1.0,1.1)]
        else:
            self.num_groups_sample_intervals = num_groups
        if sample_intervals == None:
            self.sample_intervals = [(0.1, 2)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        self.keep_ignorred = keep_ignorred

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"scale": self.sample_intervals,
            "num_groups": self.num_groups_sample_intervals}
        )

    def _augment(self, sample: Sample) -> Sample:
        for i in range(int(self.num_groups)):
            img, mask = super().transparentOverlay(
                sample["image"],
                sample["ood_mask"],
                scale=self.scale,
                mask_threshold=self.mask_threshold,
                overlay_path=join(
                    data_folder,
                    f"group/{random.choice(listdir(join(data_folder,'group')))}",
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
