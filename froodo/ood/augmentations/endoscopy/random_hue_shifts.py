from os import listdir
import os
from os.path import join
import random
from froodo.data.samples.sample import Sample
from froodo.ood.augmentations.types import OODAugmentation

import imageio
import cv2 as cv
import numpy as np
import torch

import colorsys
import scipy

import torchvision

from froodo.ood.severity.severity import ParameterSeverityMeasurement


class RandomHueShiftAugmentation(OODAugmentation):
    def __init__(self) -> None:
        super().__init__()
        self.EPS = 0.0001
        self.severity_class = ParameterSeverityMeasurement("amount", (0, 0.5))

    def _augment(self, sample: Sample) -> Sample:

        img = sample.image

        # find hue_factor in (-0.5, 0.5)
        hue_factor = np.random.normal(0, 0.2)
        hue_factor = max(min(hue_factor, 0.5), -0.5)

        self.amount = np.abs(hue_factor)

        img = torchvision.transforms.functional.adjust_hue(img, hue_factor)

        sample["image"] = img
        sample["ood_mask"] = torch.zeros_like(sample["ood_mask"])
        return sample

    def _get_parameter_dict(self):
        return {"amount": self.amount}
