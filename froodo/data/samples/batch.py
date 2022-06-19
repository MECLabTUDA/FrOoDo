from abc import abstractmethod
import torch
from torch.utils.data import _utils

from typing import List

from .sample import Sample
from ..metadata import SampleMetdadataBatch


class Batch(Sample):
    def __init__(
        self,
        images: torch.Tensor,
        labels_dict: dict,
        metadata_batch: SampleMetdadataBatch = None,
    ) -> None:
        super().__init__(images, labels_dict, metadata_batch)

    @staticmethod
    def init_from_samples(samples: List[Sample]):
        img = []
        labels = []
        meta = SampleMetdadataBatch([])
        for sample in samples:
            img.append(sample.image)
            labels.append(sample.label_dict)
            meta.batch.append(sample.metadata)
        return Batch(
            _utils.collate.default_collate(img),
            _utils.collate.default_collate(labels),
            meta,
        )
