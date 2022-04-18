import torch
from torchvision.transforms import ColorJitter

import random

from ...augmentations import OODAugmantation, ProbabilityAugmentation, PickNComposite


class BrightnessAugmentation(OODAugmantation):
    def __init__(self, brightness=1.5, prob=1) -> None:
        super().__init__()
        self.brightness = brightness
        self.prob = prob

    @property
    def brightness(self):
        return self._brightness

    @brightness.setter
    def brightness(self, value):
        if type(value) == tuple:
            self._brightness = value
        elif type(value) == float or type(value) == int:
            assert value > 0
            self._brightness = (value, value)
        self.jitter = ColorJitter(brightness=self.brightness)

    def __call__(self, img, mask):
        if random.random() >= self.prob:
            return img, mask
        return self.jitter(img), torch.zeros_like(mask)


class HigherOrLowerBrightnessAugmentation(OODAugmantation):
    def __init__(self, lower_range=(0.2, 0.6), higher_range=(1.4, 2), prob=1) -> None:
        super().__init__()
        self.augmentation = ProbabilityAugmentation(
            PickNComposite(
                [
                    BrightnessAugmentation(brightness=lower_range, prob=1),
                    BrightnessAugmentation(brightness=higher_range, prob=1),
                ]
            ),
            prob,
        )

    def __call__(self, img, mask):
        return self.augmentation(img, mask)
