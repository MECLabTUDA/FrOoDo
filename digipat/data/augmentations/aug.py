import torch
import random as random


class OODAugmantation:
    def __call__(self, img, mask):
        raise NotImplementedError("Please Implement this method")

    def do_random(self, img, mask, augmentation):
        if random.random() >= self.prob:
            return img, mask
        return augmentation(img), torch.zeros_like(mask)


class AugmentationComposite(OODAugmantation):
    def __init__(self, augmantations) -> None:
        super().__init__()
        for a in augmantations:
            assert isinstance(a, OODAugmantation)
        self.augmantations = augmantations
