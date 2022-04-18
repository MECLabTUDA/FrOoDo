import random
from os import listdir
from os.path import join

from ...augmentations import OODAugmantation
from .artifacts import ArtifactAugmentation, data_folder


class SquamousAugmentation(OODAugmantation, ArtifactAugmentation):
    def __init__(self, scale=1, path=None, mask_threshold=0.3) -> None:
        super().__init__()
        self.scale = scale
        self.path = path
        self.mask_threshold = mask_threshold

    def squamous_preproces(self, img, mask):
        return super().transparentOverlay(
            img,
            mask,
            scale=self.scale,
            overlay_path=join(
                data_folder,
                f"squamous/{random.choice(listdir(join(data_folder,'squamous')))}",
            )
            if self.path == None
            else self.path,
            mask_threshold=self.mask_threshold,
            width_slack=(-0.1, -0.1),
            height_slack=(-0.1, -0.1),
        )

    def __call__(self, img, mask):
        return self.squamous_preproces(img, mask)
