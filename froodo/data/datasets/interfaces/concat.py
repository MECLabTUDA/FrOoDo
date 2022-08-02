from torch.utils.data import Dataset, ConcatDataset

from typing import List

from . import SampleDataset
from ...samples import Sample
from ....ood.augmentations import Augmentation


class ConcatSampleDataset(Dataset, SampleDataset):
    def __init__(
        self, datasets: List[SampleDataset], transform: Augmentation = None
    ) -> None:
        super(ConcatSampleDataset, self).__init__()
        self.common_label_set = None
        for d in datasets:
            sample: Sample = d[0]
            label_set = sample.label_dict.keys()
            if self.common_label_set == None:
                self.common_label_set = set(label_set)
            else:
                self.common_label_set = self.common_label_set.intersection(
                    set(label_set)
                )
        self.transform = transform
        self.dataset = ConcatDataset(datasets)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Sample:
        sample: Sample = self.dataset[index]
        unwanted = set(sample.label_dict.keys()) - set(self.common_label_set)
        for unwanted_key in unwanted:
            del sample.label_dict[unwanted_key]
        if self.transform != None:
            return self.transform(sample)
        return sample
