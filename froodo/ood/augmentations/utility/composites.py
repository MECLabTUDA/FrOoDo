import numpy as np

from ..types import OODAugmentation
from ...augmentations import AugmentationComposite
from ....data import Sample
from ....data import SampleMetadataCommonTypes


class PickNComposite(AugmentationComposite):
    def __init__(
        self, augmentations, n=1, replace=False, probabilities=None, keep_severity=False
    ) -> None:
        super().__init__(augmentations)
        self.n = n
        self.replace = replace
        self.keep_severity = keep_severity
        if probabilities == None:
            self.probabilities = np.ones(len(self.augmentations)) / len(
                self.augmentations
            )
        else:
            self.probabilities = probabilities

    def _set_metadata(self, sample: Sample) -> Sample:
        if not self.keep_severity:
            sample.metadata[SampleMetadataCommonTypes.OOD_SEVERITY.name] = None
        return sample

    def _augment(self, sample: Sample) -> Sample:
        for augmentation in np.random.choice(
            self.augmentations, self.n, replace=self.replace, p=self.probabilities
        ):
            sample = augmentation(sample)
        return sample
