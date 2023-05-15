import numpy as np

from copy import deepcopy

from ...ood.severity import SeverityMeasurement
from ...data.datatypes import DistributionSampleType, OODReason
from ...data.metadata import SampleMetadataCommonTypes
from ...data.samples import Sample
from .utils import init_augmentation
from ...utils import sample_from_intervals


class Augmentation:
    def _augment(self, sample: Sample) -> Sample:
        raise NotImplementedError("Please Implement this method")

    def _set_metadata(self, sample: Sample) -> Sample:
        """Function to modify metadata after augmenting

        Parameters
        ----------
        sample : Sample
            Sample that metadata should be modified

        Returns
        -------
        Sample
            Sample with modified metadata
        """
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


class OODAugmentation(Augmentation):
    def _get_parameter_dict(self):
        return {"scale": self.scale}

    def _set_severity(self, sample: Sample) -> SeverityMeasurement:
        # set severity measurement
        severity: SeverityMeasurement = deepcopy(self.severity_class)
        severity.calculate_measurement(
            sample["image"], sample["ood_mask"], self._get_parameter_dict()
        )
        return severity

    def _set_metadata(self, sample: Sample) -> Sample:

        is_ood = True

        if hasattr(self, "severity_class"):
            severity = self._set_severity(sample)
            if severity.get_bin(ignore_true_bin=True) == -1:
                is_ood = False
                sample["metadata"][SampleMetadataCommonTypes.OOD_SEVERITY.name] = None
                sample["metadata"].type = DistributionSampleType.IN_DATA
            else:
                sample["metadata"][
                    SampleMetadataCommonTypes.OOD_SEVERITY.name
                ] = severity

        if is_ood:
            sample["metadata"].type = DistributionSampleType.OOD_DATA
            sample["metadata"][
                SampleMetadataCommonTypes.OOD_REASON.name
            ] = OODReason.AUGMENTATION_OOD
            sample["metadata"][SampleMetadataCommonTypes.OOD_AUGMENTATION.name] = type(
                self
            ).__name__

        return sample


class AugmentationComposite(Augmentation):
    def __init__(self, augmentations) -> None:
        super().__init__()
        for a in augmentations:
            assert isinstance(a, Augmentation)
        self.augmentations = augmentations


class INAugmentation(Augmentation):
    def _set_metadata(self, sample: Sample) -> Sample:
        return sample

    def __call__(self, sample: Sample) -> Sample:
        sample = init_augmentation(sample, create_ood=False)
        sample = self._augment(sample)
        sample = self._set_metadata(sample)
        return sample


class SizeAugmentation(Augmentation):
    def _min_max_change(self):
        raise NotImplementedError("Please Implement this method")


class SampableAugmentation(Augmentation):
    def _set_attr_to_uniform_samples_from_intervals(self, param_dict, to_int=False):
        for key, intervals in param_dict.items():
            s = sample_from_intervals(intervals)
            setattr(self, key, s if not to_int else int(s))

    def _apply_sampling(self):
        raise NotImplementedError("Please Implement this method")
