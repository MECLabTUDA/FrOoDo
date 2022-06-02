from typing import List

from ...data.datasets.interfaces import SampleDataset
from .strategy import OODStrategy


class ClassLeftoutStrategy(OODStrategy):
    def __init__(self, dataset: SampleDataset, leftout_classes: List[int]) -> None:
        super().__init__()
        self.leftout_classes = leftout_classes
        self.dataset = dataset
