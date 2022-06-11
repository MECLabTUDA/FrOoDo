import torch
import numpy as np
import cv2 as cv
from scipy import ndimage

import imageio
from random import randint
import os

from .....data.metadata import *

this_dir, this_filename = os.path.split(__file__)
data_folder = os.path.join(this_dir, "imgs")


class ArtifactAugmentation:
    def transparentOverlay(
        self,
        src,
        mask,
        scale=2,
        mask_threshold=0.3,
        overlay_path="imgs/dark_spots/small_spot.png",
        width_slack=(0.3, 0.3),
        height_slack=(0.3, 0.3),
        ignore_index=None,
    ):
        src = src.permute(1, 2, 0)
        overlay = imageio.imread(overlay_path) / 255.0
        overlay = cv.resize(overlay, (0, 0), fx=scale, fy=scale)
        index = np.round(ndimage.measurements.center_of_mass(overlay[..., 3])).astype(
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
        pos = (he - index[0], wi - index[1])

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
