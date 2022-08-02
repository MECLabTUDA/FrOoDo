from torch.utils.data import Dataset
import numpy as np

import random


from ...samples import Sample
from ...datatypes import DistributionSampleType
from ..adapter.adapter import DatasetAdapter
from .sample_dataset import SampleDataset


class OODDataset(Dataset, SampleDataset):
    def __init__(self, dataset: SampleDataset, num=None) -> None:
        super().__init__()
        assert isinstance(dataset, SampleDataset)
        self.dataset = dataset
        self.num = num

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Sample:
        sample = self.dataset.__getitem__(index)
        sample.metadata.type = DistributionSampleType.OOD_DATA
        sample.metadata["OOD_REASON"] = "OOD_DATASET"
        sample.metadata["OOD_DATASET"] = (
            type(self.dataset) if self.num == None else self.num
        )
        return sample
