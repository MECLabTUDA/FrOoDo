from torch.utils.data import DataLoader

from ..augmentations import Augmantation
from ...data.datasets.interfaces import OODAugmentationDataset
from ...data.utils import sample_collate
from ...data.datasets.interfaces import SampleDataset
from .strategy import OODStrategy


class AugmentationStrategy(OODStrategy):
    def __init__(
        self, dataset: OODAugmentationDataset, augmentation: Augmantation, seed=None
    ) -> None:
        assert isinstance(dataset, SampleDataset)
        assert isinstance(augmentation, Augmantation)
        self.dataset = OODAugmentationDataset(dataset, augmentation, seed)
