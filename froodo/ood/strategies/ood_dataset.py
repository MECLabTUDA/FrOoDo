from typing import List

from .strategy import OODStrategy
from ...data.datasets import SampleDataset
from ...data.datasets.interfaces import OODDataset, ConcatSampleDataset


class OODDatasetsStrategy(OODStrategy):
    def __init__(
        self,
        in_dataset_list: List[SampleDataset],
        ood_dataset_list: List[SampleDataset],
    ) -> None:
        super().__init__()
        self.dataset = ConcatSampleDataset(
            in_dataset_list
            + [
                OODDataset(d, d.find_base_name(default=i))
                for i, d in enumerate(ood_dataset_list)
            ]
        )
