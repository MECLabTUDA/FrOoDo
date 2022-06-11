from torch import Tensor
from skimage.filters import gaussian

from typing import Tuple
import random


from ....augmentations import OODAugmentation, SizeAugmentation, SampableAugmentation
from .....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from .....data.metadata import *
from ...utils import *
from .....data.samples import Sample


class OverlapAugmentation(OODAugmentation, SizeAugmentation, SampableAugmentation):
    def __init__(
        self,
        overlap=50,
        sigma=1.5,
        shuffle=3,
        severity: SeverityMeasurement = None,
        sample_intervals=None,
    ) -> None:
        super().__init__()
        self.overlap = overlap
        self.sigma = sigma
        self.shuffle = shuffle
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        if sample_intervals == None:
            self.sample_intervals = [(10, 100)]
        else:
            self.sample_intervals = sample_intervals

    def _get_parameter_dict(self):
        return {"overlap": self.overlap, "shuffle": self.shuffle, "sigma": self.sigma}

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"overlap": self.sample_intervals}, True
        )

    def _augment(self, sample: Sample) -> Tuple[Tensor, Tensor]:

        per_img = sample.image.permute(1, 2, 0)

        flip = random.uniform(0, 1) > 0.5

        start = int(
            random.uniform(
                self.shuffle,
                per_img.shape[0] - 2 * self.overlap
                if not flip
                else per_img.shape[1] - (2 * self.overlap),
            )
        )

        part1 = torch.ones((per_img.shape[0] - self.overlap, per_img.shape[1], 3))
        part2 = torch.ones((per_img.shape[0] - self.overlap, per_img.shape[1], 3))

        if flip:
            per_img = per_img.permute(1, 0, 2)

        part1[start:, ...] = per_img[start + self.overlap :, ...]
        part2[: start + self.overlap, ...] = per_img[: start + self.overlap, ...]

        result = part1 * part2
        shuffle1 = result[start - self.shuffle : start + self.shuffle, ...]
        shuffle1 = gaussian(shuffle1, sigma=self.sigma, multichannel=True)
        result[start - self.shuffle : start + self.shuffle, ...] = torch.from_numpy(
            shuffle1
        )

        shuffle2 = result[
            start + self.overlap - self.shuffle : start + self.overlap + self.shuffle,
            ...,
        ]
        shuffle2 = gaussian(shuffle2, sigma=self.sigma, multichannel=True)
        result[
            start + self.overlap - self.shuffle : start + self.overlap + self.shuffle,
            ...,
        ] = torch.from_numpy(shuffle2)

        if flip:
            result = result.permute(1, 0, 2)
            part1 = part1.permute(1, 0, 2)
            part2 = part2.permute(1, 0, 2)
            per_img = per_img.permute(1, 0, 2)
            if sample["ood_mask"] != None:
                sample["ood_mask"][:, start : start + self.overlap] = 0
                sample["ood_mask"] = sample["ood_mask"][:, : -self.overlap]
                # dummy
                sample["segmentation_mask"] = torch.ones_like(sample["ood_mask"])
        else:
            if sample["ood_mask"] != None:
                sample["ood_mask"][start : start + self.overlap :] = 0
                sample["ood_mask"] = sample["ood_mask"][: -self.overlap, :]
                # dummy
                sample["segmentation_mask"] = torch.ones_like(sample["ood_mask"])

        sample.image = result.permute(2, 0, 1)

        return sample
