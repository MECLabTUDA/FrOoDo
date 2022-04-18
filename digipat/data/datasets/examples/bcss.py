from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize

from ..interfaces import (
    OODAugmentationDataset,
    UtitlityDataset,
    MultiFileOverlappingTilesDataset,
)
from ...augmentations import OODAugmantation


class BCSS_Base_Dataset(MultiFileOverlappingTilesDataset, UtitlityDataset):
    def __init__(self, resize_size=(300, 300)) -> None:
        super().__init__()
        self.resize = Resize(self.resize_size, interpolation=InterpolationMode.NEAREST)

    def __getitem__(self, index):
        img, mask = super().__getitem__(index)
        img = self.resize(img.unsqueeze(0)).squeeze()
        mask = self.resize(mask.unsqueeze(0).unsqueeze(0)).squeeze()
        return img, mask


class BCSS_OOD_Dataset(OODAugmentationDataset):
    def __init__(self, augmentation: OODAugmantation) -> None:
        super().__init__(BCSS_Base_Dataset(), augmentation)
