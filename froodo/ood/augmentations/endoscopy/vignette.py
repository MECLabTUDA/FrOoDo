
import torch
from froodo.data.samples.sample import Sample
from froodo.ood.augmentations.types import INAugmentation

import numpy as np


class Vignette(INAugmentation):
    def __init__(self, percent_vignetting: float= 0) -> None:
        assert 0<= percent_vignetting <=1
        self.p = 1 - percent_vignetting
        
        #https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    def create_circular_mask(self, h, w, center=None, radius=None):

        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return torch.tensor(mask)

    def _augment(self, sample: Sample) -> Sample:
        img = sample.image.permute(1,2,0)   #CHW -> HWC
        ood_mask = sample['ood_mask']

        H,W,_ = img.shape

        mask = self.create_circular_mask(img.shape[0],img.shape[1], radius=self.p * max(H,W) / 2) 
        img[mask== 0] = torch.tensor([0,0,0]).float()
        ood_mask[mask == 0] = 1     #in distribution

        sample.image = img.permute(2,0,1)    #HWC -> CHW
        return sample
