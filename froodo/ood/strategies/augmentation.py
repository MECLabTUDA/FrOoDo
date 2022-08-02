from torch.utils.data import DataLoader

from ..augmentations import Augmentation
from ...data.datasets.interfaces import AugmentationDataset
from ...data.datasets.interfaces import SampleDataset
from .strategy import OODStrategy


class AugmentationStrategy(OODStrategy):
    def __init__(
        self, dataset: AugmentationDataset, augmentation: Augmentation, seed=None
    ) -> None:
        assert isinstance(dataset, SampleDataset)
        assert isinstance(augmentation, Augmentation)
        self.dataset = AugmentationDataset(dataset, augmentation, seed)
