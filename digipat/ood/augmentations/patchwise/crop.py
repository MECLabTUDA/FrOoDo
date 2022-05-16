import torch
from torchvision.transforms import RandomCrop, Resize

import random

from ...augmentations import OODAugmantation


class CropAugmentation(OODAugmantation):
    def __init__(self, crop=(150, 150), prob=1) -> None:
        super().__init__()
        self.prob = prob
        self.crop = crop

    @property
    def crop(self):
        return self._crop

    @crop.setter
    def crop(self, value):
        if type(value) == tuple:
            self._crop = value
        elif type(value) == float or type(value) == int:
            assert value >= 0 and value <= 1
            self._crop = (round(value * 300), round(value * 300))

    def random_crop(self, img, mask, prob, crop):
        if random.random() >= prob:
            return img, mask
        cropping = RandomCrop(size=crop)
        resize = Resize(
            (300, 300),
        )
        return resize(cropping(img)), torch.zeros_like(mask)

    def __call__(self, img, mask):
        return self.random_crop(img, mask, self.prob, self.crop)
