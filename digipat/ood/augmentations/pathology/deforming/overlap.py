from torch import Tensor
from skimage.filters import gaussian

from typing import Tuple
import random


from ....augmentations import OODAugmantation, SizeAugmentation
from .....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from .....data.metadata import *
from ...utils import *
from .....data.samples import Sample


class OverlapAugmentation(OODAugmantation, SizeAugmentation):
    def __init__(
        self,
        scale=1,
        severity: SeverityMeasurement = None,
        sample_range=None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        if sample_range == None:
            self.sample_range = {"scale": (0.1, 5)}
        else:
            self.sample_range = sample_range

    def param_range(self):
        return self.sample_range

    def _augment(self, sample: Sample) -> Tuple[Tensor, Tensor]:

        per_img = sample.image.permute(1, 2, 0)

        overlay = int(random.uniform(10, 200))
        # overlay = 10
        shuffle = 3
        sigma = 1.5

        flip = random.uniform(0, 1) > 0.5

        start = int(
            random.uniform(
                shuffle,
                per_img.shape[0] - 2 * overlay
                if not flip
                else per_img.shape[1] - (2 * overlay),
            )
        )

        part1 = torch.ones((per_img.shape[0] - overlay, per_img.shape[1], 3))
        part2 = torch.ones((per_img.shape[0] - overlay, per_img.shape[1], 3))

        if flip:
            per_img = per_img.permute(1, 0, 2)

        part1[start:, ...] = per_img[start + overlay :, ...]
        part2[: start + overlay, ...] = per_img[: start + overlay, ...]

        result = part1 * part2
        shuffle1 = result[start - shuffle : start + shuffle, ...]
        shuffle1 = gaussian(shuffle1, sigma=sigma, multichannel=True)
        result[start - shuffle : start + shuffle, ...] = torch.from_numpy(shuffle1)

        shuffle2 = result[start + overlay - shuffle : start + overlay + shuffle, ...]
        shuffle2 = gaussian(shuffle2, sigma=sigma, multichannel=True)
        result[
            start + overlay - shuffle : start + overlay + shuffle, ...
        ] = torch.from_numpy(shuffle2)

        if flip:
            result = result.permute(1, 0, 2)
            part1 = part1.permute(1, 0, 2)
            part2 = part2.permute(1, 0, 2)
            per_img = per_img.permute(1, 0, 2)
            if sample["ood_mask"] != None:
                sample["ood_mask"][:, start : start + overlay] = 0
                sample["ood_mask"] = sample["ood_mask"][:, :-overlay]
                # dummy
                sample["segmentation_mask"] = torch.ones_like(sample["ood_mask"])
        else:
            if sample["ood_mask"] != None:
                sample["ood_mask"][start : start + overlay :] = 0
                sample["ood_mask"] = sample["ood_mask"][:-overlay, :]
                # dummy
                sample["segmentation_mask"] = torch.ones_like(sample["ood_mask"])

        sample.image = result.permute(2, 0, 1)

        return sample
