import random
from os import listdir
from os.path import join
import torch
import numpy as np
import cv2 as cv
from scipy import ndimage

import imageio
from random import randint
import os

from ....augmentations import OODAugmentation, SampableAugmentation
from .....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from .....data.samples import Sample

this_dir, this_filename = os.path.split(__file__)
data_folder = os.path.join(this_dir, "imgs")


class TubesAugmentation(OODAugmentation, SampableAugmentation):
    def __init__(
        self,
        scale=1,
        path=None,
        severity: SeverityMeasurement = None,
        mask_threshold=0.3,
        sample_intervals=None,
        keep_ignored=True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.path = path
        self.mask_threshold = mask_threshold
        if sample_intervals is None:
            self.sample_intervals = [(0.3, 0.31)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity is None else severity
        )
        self.keep_ignored = keep_ignored

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"scale": self.sample_intervals}
        )

    def _augment(self, sample: Sample) -> Sample:
        img, mask = self.transparentOverlay(
            sample["image"],
            sample["ood_mask"],
            scale=self.scale,
            overlay_path=join(
                data_folder,
                f"tubes/{random.choice(listdir(join(data_folder,'tubes')))}",
            ) if self.path is None else self.path,
            mask_threshold=self.mask_threshold,
            width_slack=(0, 0),
            height_slack=(0, 0),
            ignore_index=None if not self.keep_ignored else sample.metadata["ignore_index"],
        )
        sample["image"] = img
        sample["ood_mask"] = mask

        return sample

    def transparentOverlay(
        self,
        src,
        mask,
        scale=2,
        mask_threshold=0.3,
        overlay_path="imgs/tubes/test.png",
        width_slack=(0.3, 0.3),
        height_slack=(0.3, 0.3),
        ignore_index=None,
    ):
        src = src.permute(1, 2, 0)
        overlay = imageio.imread(overlay_path) / 255.0
        overlay = cv.resize(overlay, (0, 0), fx=scale, fy=scale)
        center_of_mass = np.round(ndimage.measurements.center_of_mass(overlay[..., 3])).astype(
            np.int64
        )
        h_overlay, w_overlay, _ = overlay.shape
        h_img, w_img, _ = src.shape

        min_vert = -int(h_overlay * height_slack[0])
        max_vert = max(min_vert + 1, h_img + int(h_overlay * height_slack[1]))
        he = randint(
            min_vert,
            max_vert,
        )

        min_hor = -int(w_overlay * width_slack[0])
        max_hor = max(min_hor + 1, w_img + int(w_overlay * width_slack[1]))
        wi = randint(min_hor, max_hor)

        he_rot = randint(0, h_img - h_overlay)
        wi_rot = 0

        wi = randint(0, w_img - w_overlay)
        he = (h_overlay // 2)

        # move artifact only along axis perpendicular to tube direction
        if "rotated" in os.path.basename(overlay_path):
            pos = (he_rot - center_of_mass[0], -center_of_mass[1])
        else:
            pos = (center_of_mass[0], wi - center_of_mass[1])

        # TODO: for tube artifacts: tube should extend to the edge of the overlay and don't apply any translation in the tube direction
        # TODO: translation in other direction should be limited to stay within torso
        # TODO: add rotation (in 90 degree steps, so we don't need separate tube pictures)
        # TODO: severity should increase amount of overlayed tubes

        # check for out of bounds artifact
        assert pos[0] + h_overlay > 0 and pos[0] - h_overlay - h_img < 0
        assert pos[1] + w_overlay > 0 and pos[1] - w_overlay - w_img < 0

        from_y_art = max(-pos[0], 0)
        from_x_art = max(-pos[1], 0)
        from_y = -min(-pos[0], 0)
        from_x = -min(-pos[1], 0)
        until_y = min(h_overlay - from_y_art + from_y, h_img)
        until_x = min(w_overlay - from_x_art + from_x, w_img)
        until_y_art = from_y_art + until_y - from_y
        until_x_art = from_x_art + until_x - from_x

        print("--------------------------------------")
        print(center_of_mass)
        print(pos)
        print((from_y_art, from_x_art), (until_y_art, until_x_art))

        alpha = torch.from_numpy(overlay[:, :, 3])
        overlayed = (
            alpha[from_y_art:until_y_art, from_x_art:until_x_art, np.newaxis]
            * overlay[from_y_art:until_y_art, from_x_art:until_x_art, :3]
            + (1 - alpha[from_y_art:until_y_art, from_x_art:until_x_art, np.newaxis])
            * src[from_y:until_y, from_x:until_x, :]
        )
        src[from_y:until_y, from_x:until_x, :] = overlayed

        ood_indices = torch.from_numpy(
            overlay[from_y_art:until_y_art, from_x_art:until_x_art, 3] > mask_threshold
        )

        if ignore_index != None:
            ignore_mask = mask == ignore_index

        mask[from_y:until_y, from_x:until_x][ood_indices] = 0

        if ignore_index != None:
            mask[ignore_mask] = ignore_index

        return src.permute(2, 0, 1), mask
