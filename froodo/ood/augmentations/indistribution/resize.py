from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
from torch import Tensor

from typing import Tuple

from ..types import INAugmentation
from ....data.samples import Sample


class InResize(INAugmentation):
    def __init__(self, resize_size: Tuple[int, int]) -> None:
        super().__init__()
        self.resize_size = resize_size
        self.resize = Resize(self.resize_size, interpolation=InterpolationMode.NEAREST)

    def _augment(self, sample: Sample) -> Tuple[Tensor, Tensor]:
        img_shape = sample.image.shape
        sample.image = self.resize(sample.image.unsqueeze(0)).squeeze()
        required_size_for_resizing = (img_shape[1], img_shape[2])
        for k, v in sample.label_dict.items():
            if v.shape == required_size_for_resizing:
                sample[k] = self.resize(v.unsqueeze(0).unsqueeze(0)).squeeze()
        return sample
