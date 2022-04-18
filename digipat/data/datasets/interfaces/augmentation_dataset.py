from torch.utils.data import Dataset
import numpy as np

import random

from ...augmentations import OODAugmantation, Nothing


class OODAugmentationDataset(Dataset):
    def __init__(self, dataset: Dataset, augmentation: OODAugmantation = None) -> None:
        super().__init__()
        self.dataset = dataset
        self.set_augmentation(augmentation)

    def init_seeds(self, seed=42):
        np.random.seed(seed)
        self.seeds = (np.random.random(len(self)) * 10000).astype(np.int64)
        np.random.seed()

    def set_augmentation(self, augmentation: OODAugmantation):
        if augmentation == None:
            self.augmentation = Nothing()
        self.mode = "full_ood"
        assert isinstance(augmentation, OODAugmantation)
        self.augmentation = augmentation

    def _apply_augmentations(self, image, mask):
        return self.augmentation(image, mask)

    def __getitem__(self, index):
        if self.pseudo_random:
            random.seed(self.seeds[index])
            np.random.seed(self.seeds[index])
        return self._apply_augmentations(*super().__getitem__(index))

    def __len__(self):
        return len(self.dataset)
