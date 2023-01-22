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

class RandomValueShiftAugmentation(OODAugmentation):
    def __init__(self) -> None:
        super().__init__()
        self.EPS = 0.0001
        self.severity_class = ParameterSeverityMeasurement("amount", (self.EPS, 1 - self.EPS))


    def _augment(self, sample: Sample) -> Sample:
        
        img = sample.image

        #find brightness_factor in (0, infty)

        # amount in (0, 1) where 0 has the highest probability
        amount = np.abs(np.random.normal(0, 1))
        amount = max(min(amount, 1-self.EPS), 0 + self.EPS)       #clamp to (0, 1)

        self.amount = amount

        direction = np.random.randint(0,1+1)
        if direction == 0:
            brightness_factor = 1 - amount
        else:
            x = 1 - amount
            brightness_factor = 1/x

        img = torchvision.transforms.functional.adjust_brightness(img, brightness_factor)

        sample["image"] = img
        sample["ood_mask"] = torch.zeros_like(sample["ood_mask"])
        return sample


    def _get_parameter_dict(self):
        return {'amount': self.amount}
