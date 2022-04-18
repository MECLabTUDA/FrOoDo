from os.path import join

from ...augmentations import OODAugmantation
from .artifacts import ArtifactAugmentation, data_folder


class FatAugmentation(OODAugmantation, ArtifactAugmentation):
    def __init__(self, scale=1, mask_threshold=0.4) -> None:
        super().__init__()
        self.scale = scale
        self.mask_threshold = mask_threshold

    def __call__(self, img, mask):
        return super().transparentOverlay(
            img,
            mask,
            overlay_path=join(data_folder, "fat/fat.png"),
            scale=self.scale,
            mask_threshold=self.mask_threshold,
            width_slack=(-0.1, -0.1),
            height_slack=(-0.1, -0.1),
        )
