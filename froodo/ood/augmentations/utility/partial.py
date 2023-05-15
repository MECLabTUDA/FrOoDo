import torch
import random
import numpy as np

from copy import deepcopy

from ....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement

from ..types import Augmentation
from ....data.samples import Sample
from ...augmentations import (
    OODAugmentation,
)


class PartialOODAugmentaion(OODAugmentation):
    def __init__(self, base_augmentation, mode="linear", severity: SeverityMeasurement = None) -> None:
        super().__init__()
        self.base_augmentation = base_augmentation
        assert mode in ["linear", "axes"]
        self.mode = mode
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )

    def _get_parameter_dict(self):
        return {}

    def _augment(self, sample: Sample) -> Sample:

        _, h, w = sample.image.shape
        # direction determines direction of mask
        direction = random.uniform(0, 1) > 0.5

        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, h))

        # choose direction/mode in which the picture is cut
        if self.mode == "axes":
            # flip chooses axis of image
            flip = random.uniform(0, 1) > 0.5
            if direction:
                mask = (grid_x if flip else grid_y) > (
                    random.randint(0, w) if flip else random.randint(0, h)
                )
            else:
                mask = (grid_x if flip else grid_y) < (
                    random.randint(0, w) if flip else random.randint(0, h)
                )
        elif self.mode == "linear":
            # calculate random linear functions
            wx = random.random() * np.sqrt(2)
            wy = random.random() * np.sqrt(2)
            b = random.random() * np.sqrt(h * w / 4) * max(wx, wy)
            if random.random() > 0.5:
                wx *= -1
            else:
                wy *= -1

            mask = wx * (grid_x - (w / 2)) + wy * (grid_y - (h / 2)) + b > 0

        else:
            raise ValueError("Given mode is not valid")

        # apply augmentation
        copied_img = deepcopy(sample)
        copied_img = self.base_augmentation(copied_img)
        sample.image[:, mask] = copied_img.image[:, mask]
        sample["ood_mask"][mask] = copied_img["ood_mask"][mask]
        sample.metadata = copied_img.metadata

        return sample
