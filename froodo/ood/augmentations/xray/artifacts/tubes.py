import random
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import scipy
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


def crop(img):
    # lower bound for each channel
    lower = (0, 0, 0, 0)
    # upper bound for each channel
    upper = (0.05, 0.05, 0.05, 0.05)
    crop_mask = cv.inRange(img, lower, upper)

    # get bounds of background pixels
    background = np.where(crop_mask == 0)
    xmin, ymin, xmax, ymax = np.min(background[1]), np.min(background[0]), \
        np.max(background[1]), np.max(background[0])
    cropped = img[ymin:ymax, xmin:xmax]

    return cropped


class TubesAugmentation(OODAugmentation, SampableAugmentation):
    def __init__(
        self,
        amount=1,
        path=None,
        severity: SeverityMeasurement = None,
        mask_threshold=0.3,
        sample_intervals=None,
        keep_ignored=True,
    ) -> None:
        super().__init__()
        self.amount = amount
        self.scale = 0.25
        self.path = path
        self.mask_threshold = mask_threshold
        if sample_intervals is None:
            self.sample_intervals = [(1, 6)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity is None else severity
        )
        self.keep_ignored = keep_ignored

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"amount": self.sample_intervals}
        )

    @property
    def amount(self):
        return self._amount

    @amount.setter
    def amount(self, value):
        self._amount = int(value)

    def _augment(self, sample: Sample) -> Sample:
        img, mask = sample["image"], sample["ood_mask"]

        for i in range(self.amount):
            img, mask = self.transparent_overlay(
                img,
                mask,
                overlay_path=join(
                    data_folder,
                    f"tubes/{random.choice(listdir(join(data_folder,'tubes')))}",
                ) if self.path is None else self.path,
                mask_threshold=self.mask_threshold,
                ignore_index=None if not self.keep_ignored else sample.metadata["ignore_index"],
            )

        sample["image"] = img
        sample["ood_mask"] = mask
        return sample

    def transparent_overlay(
        self,
        src,
        mask,
        overlay_path,
        mask_threshold=0.3,
        ignore_index=None,
    ):
        scale = self.scale
        src = src.permute(1, 2, 0)
        overlay = imageio.imread(overlay_path) / 255.0
        overlay = cv.resize(overlay, (0, 0), fx=scale, fy=scale)

        # randomly flip artifact
        if randint(0, 1):
            overlay = np.flip(overlay, axis=1)

        base_rotation = random.choice([-90, 0, 90])
        rotation_angle = np.random.randint(-20, high=20)
        overlay = scipy.ndimage.rotate(overlay, base_rotation + rotation_angle, order=1)

        overlay = crop(overlay)

        h_img, w_img, _ = src.shape
        h_overlay, w_overlay, _ = overlay.shape

        if base_rotation == 0:
            from_y_art = 0
            until_y_art = from_y_art + h_overlay

            from_x_art = 0
            until_x_art = from_x_art + w_overlay

            x_slack = w_img - w_overlay
            from_x = randint(0, x_slack)
            until_x = from_x + w_overlay

            from_y = h_img - h_overlay
            until_y = h_img

        elif base_rotation == 90:
            from_y_art = 0
            until_y_art = from_y_art + h_overlay

            from_x_art = 0
            until_x_art = from_x_art + w_overlay

            from_x = w_img - w_overlay
            until_x = w_img

            y_slack = h_img - h_overlay
            from_y = randint(0, y_slack)
            until_y = from_y + h_overlay

        elif base_rotation == -90:
            from_y_art = 0
            until_y_art = from_y_art + h_overlay

            from_x_art = 0
            until_x_art = from_x_art + w_overlay

            from_x = 0
            until_x = from_x + w_overlay

            y_slack = h_img - h_overlay
            from_y = randint(0, y_slack)
            until_y = from_y + h_overlay
        else:
            raise "Internal Error: unsupported rotation for tubes augmentation"

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
