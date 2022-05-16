import random

from ...augmentations import OODAugmantation


class ProbabilityAugmentation(OODAugmantation):
    def __init__(self, augmentation, prob=0.5) -> None:
        super().__init__()
        assert isinstance(augmentation, OODAugmantation)
        self.augmentation = augmentation
        self.prob = prob

    def __call__(self, img, mask):
        if random.random() >= self.prob:
            return img, mask
        else:
            return self.augmentation(img, mask)


class Nothing(OODAugmantation):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, img, mask):
        return img, mask


class NTimesAugmentation(OODAugmantation):
    def __init__(self, augmentation, n) -> None:
        super().__init__()
        self.augmentation = augmentation
        self.n = n

    def __call__(self, img, mask):
        for _ in range(self.n):
            img, mask = self.augmentation(img, mask)
        return img, mask