from torch.utils.data import Dataset

from typing import List

from ...metadata.types import SampleMetadataCommonTypes
from ...samples import Sample
from ....ood.augmentations import Augmentation, Nothing
from ..adapter.adapter import DatasetAdapter
from .utility_dataset import SampleDataset
from ..adapter import DatasetAdapter
from ...datatypes import DistributionSampleType, TaskType, OODReason


class OODAugmentationDataset(Dataset, SampleDataset):
    def __init__(
        self,
        dataset: SampleDataset,
        class_leftout: List,
        task_type: TaskType.CLASSIFICATION,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.class_leftout = class_leftout
        self.task_type = task_type

    def __getitem__(self, index):
        sample: Sample = self.dataset.__getitem__(index)
        if self.task_type is TaskType.CLASSIFICATION:
            if sample["class_vector"] != None:
                pass
            elif sample["class_number"] != None:
                if sample["class_number"] in self.class_leftout:
                    sample.metadata.type = DistributionSampleType.OOD_DATA
                    sample.metadata[
                        SampleMetadataCommonTypes.OOD_REASON.name
                    ] = OODReason.UNSEEN_CLASSES_OOD.name
            else:
                raise NotImplementedError()
        elif self.task_type is TaskType.SEGMENTATION:
            pass

    def __len__(self):
        return len(self.dataset)
