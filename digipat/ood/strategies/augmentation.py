from torch.utils.data import DataLoader

from ..augmentations import Augmantation
from ...data.datasets.interfaces import OODAugmentationDataset
from ...data.utils import sample_collate
from ...data.datasets.adapter import DatasetAdapter
from .strategy import OODStrategy


class AugmentationStrategy(OODStrategy):
    def __init__(
        self, dataset: OODAugmentationDataset, augmentation: Augmantation, seed=None
    ) -> None:
        assert isinstance(dataset, DatasetAdapter)
        assert isinstance(augmentation, Augmantation)
        self.dataset = OODAugmentationDataset(dataset, augmentation, seed)

    def get_dataloader(self, **dataloader_kwargs):
        return DataLoader(self.dataset, collate_fn=sample_collate, **dataloader_kwargs)
