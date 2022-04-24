from enum import Enum, auto


class DistributionSampleType(Enum):
    IN_DISTRIBUTION_DATA = auto()
    AUGMENTATION_OOD_DATA = auto()
    DATASET_OOD_DATA = auto()
