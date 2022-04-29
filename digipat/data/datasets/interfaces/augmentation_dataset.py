from torch.utils.data import Dataset
import numpy as np

import random

from ...augmentations import OODAugmantation, Nothing
from ...datatypes import TaskType


class OODAugmentationDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        augmentation: OODAugmantation = None,
        task_type: TaskType = TaskType.SEGMENTATION,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.set_augmentation(augmentation)
        self.pseudo_random = False
        self.task_type = task_type

    def init_seeds(self, seed=42):
        self.pseudo_random = True
        np.random.seed(seed)
        self.seeds = (np.random.random(len(self)) * 10000).astype(np.int64)
        np.random.seed()

    def set_augmentation(self, augmentation: OODAugmantation):
        if augmentation == None:
            self.augmentation = Nothing()
            return
        self.dataset.mode = "full_ood"
        assert isinstance(augmentation, OODAugmantation)
        self.augmentation = augmentation

    def _apply_augmentations(self, *args):
        if self.task_type is TaskType.SEGMENTATION:
            return self.augmentation(*args)
        elif self.task_type is TaskType.CLASSIFICATION:
            # In case of classification the label is not a mask. Therefore for the augmentation a pseudo_mask is created to calculate sevrity
            aug_args = args
            aug_args[1] = None
            _img, _, _metadata = self.augmentation(*aug_args)
            return _img, args[1], _metadata

    def __getitem__(self, index):
        if self.pseudo_random:
            random.seed(self.seeds[index])
            np.random.seed(self.seeds[index])
        return self._apply_augmentations(*self.dataset.__getitem__(index))

    def __len__(self):
        return len(self.dataset)
