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

class BloodAugmentation(OODAugmentation):
    def __init__(self) -> None:
        super().__init__()
        self.severity_class = ParameterSeverityMeasurement("saturation_factor", (1,1.3))


    def _augment(self, sample: Sample) -> Sample:
        img = sample.image
        img = torchvision.transforms.functional.adjust_hue(img, -20/180)

        #find saturation_factor in [1,1.3]
        saturation_factor = float(np.random.randint(100, high=130+1))
        saturation_factor /= 100

        self.saturation_factor = saturation_factor

        img = torchvision.transforms.functional.adjust_saturation(img, saturation_factor)

        sample["image"] = img
        sample["ood_mask"] = torch.zeros_like(sample["ood_mask"])
        return sample


    def _get_parameter_dict(self):
        return {'saturation_factor': self.saturation_factor}
