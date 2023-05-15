import random
from os import listdir
from os.path import join

from typing import Optional, Tuple


from ....augmentations import OODAugmentation, SampableAugmentation
from .artifacts import ArtifactAugmentation, data_folder
from .....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from .....data.metadata import *
from ...utils import *
from .....data.samples import Sample


class BloodCellAugmentation(
    ArtifactAugmentation, OODAugmentation, SampableAugmentation
):
    def __init__(
        self,
        num_cells: int = 10,
        scale=1,
        severity: SeverityMeasurement = None,
        path=None,
        mask_threshold: float = 0.5,
        sample_intervals: Optional[List[Tuple[float, float]]] = None,
        scale_sample_intervals: Optional[List[Tuple[float, float]]] = None,
        keep_ignorred: bool = True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.num_cells = num_cells
        self.path = path
        self.mask_threshold = mask_threshold
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        if sample_intervals == None:
            self.sample_intervals = [(1, 50)]
        else:
            self.sample_intervals = sample_intervals

        self.scale_sample_intervals = scale_sample_intervals
        self.keep_ignorred = keep_ignorred

    def _apply_sampling(self):
        super()._set_attr_to_uniform_samples_from_intervals(
            {"num_cells": self.sample_intervals}
        )

    def _augment(self, sample: Sample) -> Sample:
        for i in range(int(self.num_cells)):
            if self.scale_sample_intervals != None:
                super()._set_attr_to_uniform_samples_from_intervals(
                    {"scale": self.scale_sample_intervals}
                )
            img, mask = super().transparentOverlay(
                sample["image"],
                sample["ood_mask"],
                scale=self.scale,
                mask_threshold=self.mask_threshold,
                overlay_path=join(
                    data_folder,
                    f"cell/{random.choice(listdir(join(data_folder,'cell')))}",
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
