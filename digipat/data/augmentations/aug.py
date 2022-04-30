import torch
from torch import Tensor, tensor

from typing import Tuple
import random as random
from copy import deepcopy

from ...ood.severity import SeverityMeasurement
from ..datatypes import DistributionSampleType, OODReason
from ..metadata import SampleMetadata, MetadataCommonTypes
from .utils import init_augmentation


class Augmantation:
    def _augment(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("Please Implement this method")

    def _set_metadata(
        self, img: Tensor, mask: tensor, metadata: SampleMetadata
    ) -> SampleMetadata:

        severity: SeverityMeasurement = deepcopy(self.severity_class)
        severity.calculate_measurement(img, mask, {"scale": self.scale})
        if severity.get_bin(ignore_true_bin=True) == -1:
            metadata.type = DistributionSampleType.IN_DATA
        else:
            metadata.type = DistributionSampleType.OOD_DATA
            metadata[MetadataCommonTypes.OOD_REASON.name] = OODReason.AUGMENTATION_OOD
        metadata[MetadataCommonTypes.OOD_SEVERITY.name] = severity

        return metadata

    def __call__(
        self, img: Tensor, mask: Tensor = None, metadata: SampleMetadata = None
    ) -> Tuple[Tensor, Tensor, SampleMetadata]:

        mask, metadata = init_augmentation(img, mask, metadata)

        img, mask = self._augment(img, mask)

        metadata = self._set_metadata(img, mask, metadata)

        return img, mask, metadata


class OODAugmantation(Augmantation):
    def do_random(self, img, mask, augmentation):
        if random.random() >= self.prob:
            return img, mask
        return augmentation(img), torch.zeros_like(mask)


class AugmentationComposite(OODAugmantation):
    def __init__(self, augmantations) -> None:
        super().__init__()
        for a in augmantations:
            assert isinstance(a, OODAugmantation)
        self.augmantations = augmantations


class INAugmantation(Augmantation):
    def _set_metadata(
        self, img: Tensor, mask: Tensor, metadata: SampleMetadata
    ) -> SampleMetadata:
        return metadata
