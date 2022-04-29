from torch.utils.data import DataLoader

from ...data.augmentations.aug import OODAugmantation
from ...data.datasets.interfaces import OODAugmentationDataset
from ...data.utils import augmentation_collate


class AugmentationStrategy:
    def __init__(
        self,
        augmentation_dataset: OODAugmentationDataset,
        augmentation: OODAugmantation,
    ) -> None:
        self.dataset = augmentation_dataset
        self.dataset.set_augmentation(augmentation)

    def get_dataloader(self, **dataloader_kwargs):
        return DataLoader(
            self.dataset, collate_fn=augmentation_collate, **dataloader_kwargs
        )
