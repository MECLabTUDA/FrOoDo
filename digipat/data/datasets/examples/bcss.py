from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize

from ..interfaces import (
    OODAugmentationDataset,
    UtitlityDataset,
    MultiFileOverlappingTilesDataset,
)
from ...augmentations import OODAugmantation


class BCSS_Base_Dataset(MultiFileOverlappingTilesDataset, UtitlityDataset):
    def __init__(
        self,
        folder="D:\ssl4uc\Original Data\BCSS_TIF",
        resize_size=(300, 300),
        tile_folder="D:\ssl4uc\Code\MahOoD\\tiles",
        transform=None,
        size=(620, 620),
        crop_size=(600, 600),
        dataset_name="bcss",
        mode="mask",
        overlap=0.2,
        map_classes=[],
        ignore_classes=[0, 7, 17],
        ood_classes=[5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21],
        remain_classes=[1, 2, 3, 4],
        ignore_index=5,
        non_ignore_threshold=0.9,
        files=None,
        force_overwrite=False,
    ):
        super().__init__(
            folder,
            tile_folder,
            transform,
            size,
            crop_size,
            dataset_name,
            mode,
            overlap,
            map_classes,
            ignore_classes,
            ood_classes,
            remain_classes,
            ignore_index,
            non_ignore_threshold,
            files,
            force_overwrite,
        )
        self.resize_size = resize_size
        self.resize = Resize(self.resize_size, interpolation=InterpolationMode.NEAREST)

    def __getitem__(self, index):
        img, mask = super().__getitem__(index)
        img = self.resize(img.unsqueeze(0)).squeeze()
        mask = self.resize(mask.unsqueeze(0).unsqueeze(0)).squeeze()
        return img, mask


class BCSS_OOD_Dataset(OODAugmentationDataset, UtitlityDataset):
    def __init__(
        self, bcss_base: BCSS_Base_Dataset = None, augmentation: OODAugmantation = None
    ) -> None:
        super().__init__(
            BCSS_Base_Dataset() if bcss_base == None else bcss_base, augmentation
        )
