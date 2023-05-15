from torch import Tensor
from skimage.filters import gaussian

from typing import Tuple
import random
import numpy as np
import torch


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

        # bn is the border noise e.g. number of rows/columns in which noise is applied to the partial edge
        border_noise = 0

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

        shape = np.arange(per_img.shape[0])

        # create masks to add random noise at edges
        PIMask1 = torch.full((per_img.shape[0], per_img.shape[1]), False)
        PImask2 = torch.full((per_img.shape[0], per_img.shape[1]), False)
        PIMask1[start + self.overlap :, shape] = True
        PImask2[: start + self.overlap, ...] = True

        PMask1 = torch.full((per_img.shape[0] - self.overlap, per_img.shape[1]), False)
        PMask2 = torch.full((per_img.shape[0] - self.overlap, per_img.shape[1]), False)
        PMask1[start:, shape] = True
        PMask2[: start + self.overlap, ...] = True

        # add the actual noise to the masks
        for i in range(border_noise):
            noise = torch.from_numpy(
                np.random.choice([False, True], per_img.shape[0], [0.4, 0, 6])
            )

            PIMask1[start + self.overlap + i, shape] = noise
            PImask2[start + self.overlap - i, ...] = noise
            PMask1[start + i, shape] = noise
            PMask2[start + self.overlap - i, ...] = noise

        part1[PMask1] = per_img[PIMask1]
        part2[PMask2] = per_img[PImask2]

        # part1[start:, shape] = per_img[start + self.overlap :, shape]
        # part2[:start + self.overlap, ...] = per_img[: start + self.overlap, ...]

        # combine both parts with raytracing formular
        result = part1 * part2

        # smooth edge with gaussian
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


class FoldingAugmentation(OODAugmentation, SizeAugmentation, SampableAugmentation):
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

        # bn is the border noise e.g. number of rows/columns in which noise is applied to the partial edge
        border_noise = 0

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

        shape = np.arange(per_img.shape[0])

        # create masks to add random noise at edges
        PIMask1 = torch.full((per_img.shape[0], per_img.shape[1]), False)
        PImask2 = torch.full((per_img.shape[0], per_img.shape[1]), False)
        PIMask1[start + self.overlap :, shape] = True
        PImask2[: start + self.overlap, ...] = True

        PMask1 = torch.full((per_img.shape[0] - self.overlap, per_img.shape[1]), False)
        PMask2 = torch.full((per_img.shape[0] - self.overlap, per_img.shape[1]), False)
        PMask1[start:, shape] = True
        PMask2[: start + self.overlap, ...] = True

        # add the actual noise to the masks
        for i in range(border_noise):
            noise = torch.from_numpy(
                np.random.choice([False, True], per_img.shape[0], [0.4, 0, 6])
            )

            PIMask1[start + self.overlap + i, shape] = noise
            PImask2[start + self.overlap - i, ...] = noise
            PMask1[start + i, shape] = noise
            PMask2[start + self.overlap - i, ...] = noise

        part1[PMask1] = per_img[PIMask1]
        part2[PMask2] = per_img[PImask2]

        # part1[start:, shape] = per_img[start + self.overlap :, shape]
        # part2[:start + self.overlap, ...] = per_img[: start + self.overlap, ...]
        part1[start : start + self.overlap, ...] = torch.flip(
            part1[start : start + self.overlap, ...], [0]
        )
        # part2[start:start+self.overlap,...] = torch.flip(part2[start:start+self.overlap,...],[1])

        # combine both parts with raytracing formular
        result = part1 * part2

        # smooth edge with gaussian
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
