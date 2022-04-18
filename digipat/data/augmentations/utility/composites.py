import numpy as np

from digipat.data.augmentations.aug import OODAugmantation

from ...augmentations import AugmentationComposite


class PickNComposite(AugmentationComposite):
    def __init__(
        self,
        augmantations,
        n=1,
        replace=False,
        probabilities=None,
    ) -> None:
        super().__init__(augmantations)
        self.n = n
        self.replace = replace
        if probabilities == None:
            self.probabilities = np.ones(len(self.augmantations)) / len(
                self.augmantations
            )
        else:
            self.probabilities = probabilities

    def __call__(self, img, mask):
        for augmentation in np.random.choice(
            self.augmantations, self.n, replace=self.replace, p=self.probabilities
        ):
            img, mask = augmentation(img, mask)
        return img, mask


class AllComposite(AugmentationComposite):
    def __init__(
        self,
        augmantations,
    ) -> None:
        super().__init__(augmantations)

    def __call__(self, img, mask):
        for augmentation in self.augmantations:
            img, mask = augmentation(img, mask)
        return img, mask
