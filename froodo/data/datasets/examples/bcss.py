from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize

from ..interfaces import (
    OODAugmentationDataset,
    SampleDataset,
    MultiFileOverlappingTilesDataset,
)
from ....ood.augmentations import OODAugmentation, InCrop, InResize, SizeInOODPipeline
from ..adapter.adapter import AlreadyASampleAdapter, ImageLabelMetaAdapter


class BCSS_Base_Dataset(MultiFileOverlappingTilesDataset, SampleDataset):
    def __init__(
        self,
        folder="<path to original BCSS_TIF>",
        resize_size=(300, 300),
        tile_folder="<path to tiles extracted from BCSS>",
        size=(620, 620),
        dataset_name="bcss",
        mode=["mask", "full_ood"],
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
            size,
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
        img, masks = super().__getitem__(index)
        # img = self.resize(img.unsqueeze(0)).squeeze()
        # for k, v in masks.items():
        #    masks[k] = self.resize(v.unsqueeze(0).unsqueeze(0)).squeeze()
        return img, masks


class BCSS_Adapted_Datasets:
    def __init__(self, size=(700, 700), split=(0.10, 0.20), **kwargs):
        train, val, test = BCSS_Base_Dataset(size=size).get_train_val_test_set(*split)
        self.train = ImageLabelMetaAdapter(train, ignore_index=5, **kwargs)
        self.val = ImageLabelMetaAdapter(val, ignore_index=5, **kwargs)
        self.test = ImageLabelMetaAdapter(test, ignore_index=5, **kwargs)


class BCSS_Adapted_Cropped_Resized_Datasets:
    def __init__(self, size=(700, 700), split=(0.10, 0.20), **kwargs):
        train, val, test = BCSS_Base_Dataset(size=size).get_train_val_test_set(*split)
        pipeline = SizeInOODPipeline(
            in_augmentations=[InCrop((600, 600)), InResize(resize_size=(300, 300))]
        )
        self.train = OODAugmentationDataset(
            ImageLabelMetaAdapter(train, ignore_index=5, **kwargs), pipeline
        )

        self.val = OODAugmentationDataset(
            ImageLabelMetaAdapter(val, ignore_index=5, **kwargs), pipeline
        )

        self.test = OODAugmentationDataset(
            ImageLabelMetaAdapter(test, ignore_index=5, **kwargs), pipeline
        )


class BCSS_OOD_Dataset(OODAugmentationDataset, SampleDataset):
    def __init__(
        self,
        bcss_base: BCSS_Base_Dataset = None,
        augmentation: OODAugmentation = None,
        seed=None,
    ) -> None:
        super().__init__(
            BCSS_Base_Dataset() if bcss_base == None else bcss_base, augmentation, seed
        )
