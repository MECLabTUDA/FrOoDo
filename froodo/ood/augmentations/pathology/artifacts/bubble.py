import random
from os import listdir
from os.path import join

import cv2
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import GaussianBlur
from copy import deepcopy

from ....augmentations import OODAugmentation
from .artifacts import ArtifactAugmentation, data_folder
from .....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from .....data.metadata import *
from ...utils import *
from .....data.samples import Sample


#op100, hard98
#op51, hard44 3 times!
class BubbleAugmentation(ArtifactAugmentation, OODAugmentation):
    def __init__(
        self,
        base_augmentation = GaussianBlur(kernel_size=(19, 19),sigma=5),
        scale=1,
        severity: SeverityMeasurement = None,
        path=None,
        mask_threshold=0.1,
        sample_range=None,
        keep_ignorred=True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.path = path
        self.base_augmentation = base_augmentation
        self.mask_threshold = mask_threshold
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        if sample_range == None:
            self.sample_range = {"scale": (0.5, 0.9)}
        else:
            self.sample_range = sample_range
        self.keep_ignorred = keep_ignorred

    def param_range(self):
        return self.sample_range

    def _augment(self, sample: Sample) -> Sample:

        #sample["image2"] = sample.image.clone().permute(1,2,0)

        #set up transforms
        _, h, w = sample.image.shape
        toPIL = T.ToPILImage()
        toTensor = T.ToTensor()

        #get overlay (should have alpha value)
        overlay_path=join(
                "froodo/ood/augmentations/pathology/artifacts/imgs/bubble",
                random.choice(
                    listdir("froodo/ood/augmentations/pathology/artifacts/imgs/bubble")
                ))
        overlay = Image.open(overlay_path)

        #resize overlay to image size and add to base image
        overlay = overlay.resize((h,w))
        img = toPIL(sample["image"])
        img.paste(overlay,(0,0),overlay)
        img = toTensor(img)

        #prepare further computation
        overlay = toTensor(overlay)
        copied_img = deepcopy(img)

        #coinflip on wether to apply the augmentation inside or outside of bubble
        if bool(random.getrandbits(1)) == 1:
            img[:,overlay[3,:,:] > self.mask_threshold] = self.base_augmentation(copied_img)[:,overlay[3,:,:] > self.mask_threshold]
        else:
            img[:,overlay[3,:,:] < self.mask_threshold] = self.base_augmentation(copied_img)[:,overlay[3,:,:] < self.mask_threshold]
        
        #set sample
        sample["image"] = img
        sample["ood_mask"][overlay[3,:,:] > self.mask_threshold] = 0 

        return sample