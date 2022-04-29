from torch import Tensor

from typing import Tuple
import random
from os import listdir
from os.path import join


from ...augmentations import OODAugmantation
from .artifacts import ArtifactAugmentation, data_folder
from ....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from ...metadata import *
from ..utils import *


class DarkSpotsAugmentation(ArtifactAugmentation, OODAugmantation):
    def __init__(
        self,
        scale=1,
        severity: SeverityMeasurement = None,
        path=join(data_folder, "dark_spots/small_spot.png"),
        mask_threshold=0.5,
        sample_range=None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.path = path
        self.mask_threshold = mask_threshold
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        if sample_range == None:
            self.sample_range = {"scale": (0.1, 5)}
        else:
            self.sample_range = sample_range

    def param_range(self):
        return self.sample_range

    def _augment(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return super().transparentOverlay(
            img,
            mask,
            scale=self.scale,
            mask_threshold=self.mask_threshold,
            overlay_path=join(
                data_folder,
                f"dark_spots/{random.choice(listdir(join(data_folder,'dark_spots')))}",
            )
            if self.path == None
            else self.path,
            width_slack=(-0.1, -0.1),
            height_slack=(-0.1, -0.1),
        )
