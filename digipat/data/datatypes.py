from enum import Enum, auto


class DistributionSampleType(Enum):
    UNDEFINED = auto()
    IN_DATA = auto()
    OOD_DATA = auto()


class OODReason(Enum):
    AUGMENTATION_OOD = auto()
    UNSEEN_CLASSES_OOD = auto()


class TaskType(Enum):
    SEGMENTATION = auto()
    CLASSIFICATION = auto()
