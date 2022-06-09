from torch import Tensor

from typing import Tuple
import random

from .. import INAugmentation
from ....data.samples import Sample


class InCrop(INAugmentation):
    def __init__(self, crop_size) -> None:
        super().__init__()
        self.crop_size = crop_size

    def _augment(self, sample: Sample) -> Tuple[Tensor, Tensor]:
        _, height, width = sample.image.shape
        x = random.randint(0, width - self.crop_size[1])
        y = random.randint(0, height - self.crop_size[0])
        sample.image = sample.image[
            :, y : y + self.crop_size[0], x : x + self.crop_size[1]
        ]
        required_size_for_resizing = (height, width)
        for k, v in sample.label_dict.items():
            if v.shape == required_size_for_resizing:
                sample[k] = v[y : y + self.crop_size[0], x : x + self.crop_size[1]]
        return sample
