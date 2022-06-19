from torch.utils.data import ConcatDataset

from typing import List

from .strategy import OODStrategy
from ...data.datasets.adapter import DatasetAdapter, AlreadyASampleAdapter
from ...data.datasets.interfaces import OODDataset


class OODDatasetsStrategy(OODStrategy):
    def __init__(
        self,
        in_dataset_list: List[DatasetAdapter],
        ood_dataset_list: List[DatasetAdapter],
    ) -> None:
        super().__init__()
        self.dataset = ConcatDataset(
            in_dataset_list + [OODDataset(d) for d in ood_dataset_list]
        )
