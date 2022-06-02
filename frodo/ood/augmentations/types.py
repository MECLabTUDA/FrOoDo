import torch
from torch import Tensor, tensor

from typing import Tuple
import random as random
from copy import deepcopy

from ...ood.severity import SeverityMeasurement
from ...data.datatypes import DistributionSampleType, OODReason
from ...data.metadata import SampleMetadata, SampleMetadataCommonTypes
from .utils import init_augmentation
from ...data.samples import Sample


class Augmantation:
    def _augment(self, sample: Sample) -> Sample:
        raise NotImplementedError("Please Implement this method")

    def _set_severity(self, sample: Sample) -> SeverityMeasurement:
        # set severity measurement
        severity: SeverityMeasurement = deepcopy(self.severity_class)
        severity.calculate_measurement(
            sample["image"], sample["ood_mask"], {"scale": self.scale}
        )
        return severity

    def _set_metadata(self, sample: Sample) -> Sample:

        severity = self._set_severity(sample)

        # set DistributionSampleType
        if severity.get_bin(ignore_true_bin=True) != -1:
            sample["metadata"].type = DistributionSampleType.OOD_DATA
            sample["metadata"][
                SampleMetadataCommonTypes.OOD_REASON.name
            ] = OODReason.AUGMENTATION_OOD
            sample["metadata"][SampleMetadataCommonTypes.OOD_SEVERITY.name] = severity

        return sample

    def __call__(self, sample: Sample) -> Sample:
        """Function call to augment sample

        Parameters
        ----------
        sample : Sample
            Sample that should be augmented

        Returns
        -------
        Sample
            Augmented sample
        """

        sample = init_augmentation(sample)

        sample = self._augment(sample)

        sample = self._set_metadata(sample)

        return sample


class OODAugmantation(Augmantation):
    def do_random(self, sample: Sample, augmentation):
        if random.random() >= self.prob:
            return sample
        return augmentation(sample)


class AugmentationComposite(OODAugmantation):
    def __init__(self, augmantations) -> None:
        super().__init__()
        for a in augmantations:
            assert isinstance(a, OODAugmantation)
        self.augmantations = augmantations


class INAugmantation(Augmantation):
    def _set_metadata(self, sample: Sample) -> Sample:
        return sample


class SizeAugmentation(Augmantation):
    pass
