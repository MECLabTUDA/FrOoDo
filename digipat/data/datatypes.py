from enum import Enum, auto


class DistributionSampleType(Enum):
    UNDEFINED = auto()
    IN_DISTRIBUTION_DATA = auto()
    AUGMENTATION_OOD_DATA = auto()
    DATASET_OOD_DATA = auto()


class TaskType(Enum):
    SEGMENTATION = auto()
    CLASSIFICATION = auto()
