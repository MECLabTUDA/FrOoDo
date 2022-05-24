from torch.utils.data import Dataset
import numpy as np

import random


from ...samples import Sample

from ....ood.augmentations import Augmantation, Nothing
from ..adapter.adapter import DatasetAdapter
from .utility_dataset import UtitlityDataset


class OODAugmentationDataset(Dataset, UtitlityDataset):
    def __init__(
        self, dataset: DatasetAdapter, augmentation: Augmantation = None, seed=None
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.set_augmentation(augmentation)
        if seed == None:
            self.pseudo_random = False
        else:
            self.init_seeds(seed)

    def init_seeds(self, seed=42):
        self.pseudo_random = True
        np.random.seed(seed)
        self.seeds = (np.random.random(len(self)) * 10000).astype(np.int64)
        np.random.seed()

    def set_augmentation(self, augmentation: Augmantation):
        if augmentation == None:
            self.augmentation = Nothing()
            return
        assert isinstance(augmentation, Augmantation)
        self.augmentation = augmentation

    def _apply_augmentations(self, sample: Sample):
        sample = self.augmentation(sample)
        return sample

    def __getitem__(self, index):
        if self.pseudo_random:
            random.seed(self.seeds[index])
            np.random.seed(self.seeds[index])
        return self._apply_augmentations(self.dataset.__getitem__(index))

    def __len__(self):
        return len(self.dataset)
