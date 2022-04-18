import random
from os import listdir
from os.path import join

from ...augmentations import OODAugmantation
from .artifacts import ArtifactAugmentation, data_folder


class DarkSpotsAugmentation(OODAugmantation, ArtifactAugmentation):
    def __init__(
        self,
        scale=1,
        path=join(data_folder, "dark_spots/small_spot.png"),
        mask_threshold=0.5,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.path = path
        self.mask_threshold = mask_threshold

    def __call__(self, img, mask):
        return super().transparentOverlay(
            img,
            mask,
            scale=self.scale,
            mask_threshold=self.mask_threshold,
            overlay_path=join(
                data_folder,
                f"dark_spots/{random.choice(listdir(join(data_folder,'dark_spots')))}",
            )
            if self.path == None
            else self.path,
            width_slack=(-0.1, -0.1),
            height_slack=(-0.1, -0.1),
        )
