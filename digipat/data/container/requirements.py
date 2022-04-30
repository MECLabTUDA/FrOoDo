from enum import Enum, auto


class ContainerRequirements(Enum):
    IMAGES = auto()
    MASKS = auto()
    METADATA = auto()
