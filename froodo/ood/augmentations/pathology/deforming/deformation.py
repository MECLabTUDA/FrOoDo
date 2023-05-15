
from xmlrpc.client import FastUnmarshaller
import elasticdeform.torch as ed
import numpy as np
import cv2

import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from torchvision.transforms import Resize
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as F
from torchvision.io import read_image 
import random

from ....augmentations import OODAugmentation
from .....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from .....data.metadata import *
from ...utils import *
from .....data.samples import Sample

# grad intensity = param for tweaking the intensity of the color changes, i.e. brightness
# color_thr = threshold from 0-1 for the whiteness underneath the color changes get applied. Is calulated in HLS color scheme
#0.4-0.8 for squeeze
#2.0/5.0 - 0.95
class Deformation(OODAugmentation):
    def __init__(
        self,
        scale=1,
        severity: SeverityMeasurement = None,
        path=None,
        mask_threshold=10,
        grid_points = 20,
        grad_intensity = 0.4,
        color_thr = 0.8,
        mode = "stretch",
        sample_range=None,
        keep_ignorred=True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.path = path
        self.mask_threshold = mask_threshold
        self.grid_points = grid_points
        self.grad_intensity = grad_intensity
        self.color_thr = color_thr
        self.mode = mode
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
        
        _, h, w = sample.image.shape
        max_grad = int(h/self.grid_points)
        resize = Resize((h,w))

        # #param for tweaking the intensity of the color changes, i.e. brightness
        #grad_intensity = 0.6 #0.4
        # #threshold from 0-1 for the whiteness underneath the color changes get applied. Is calulated in HLS color scheme
        #color_thr = 0.95 #0.8

        if self.mode == "stretch":
            path=join(
                "froodo/ood/augmentations/pathology/deforming/imgs/stretch",
                random.choice(
                    listdir("froodo/ood/augmentations/pathology/deforming/imgs/stretch")
                ),
            )
        elif self.mode == "squeeze":
            path=join(
                "froodo/ood/augmentations/pathology/deforming/imgs/squeeze",
                random.choice(
                    listdir("froodo/ood/augmentations/pathology/deforming/imgs/squeeze")
                ),
            )
        else:
            raise ValueError('Selected mode is invalid. Please select from "stretch" or "squeeze"')

        aug = read_image(path).int()

        #119 is grey in the test images, change with mean later maybe
        #substract mean, center around 0 and convert so size (h,w)
        aug -= 119
        aug = aug / 255 * max_grad
        aug = resize(aug)
        aug = aug[1:,:,:]
        aug[1,:,:] = 0

        # skip deformation for high intensities, when its not visible anyway
        if (self.grad_intensity < 1):
            step_h = int(h/self.grid_points)
            step_w = int(w/self.grid_points)

            grid = aug[:,::step_h,::step_w]

            sample["image2"] = sample.image.clone().permute(1,2,0)

            img = ed.deform_grid(sample["image"],grid, axis=(1,2))
        else:
            img = sample["image"]

        # prepare deformation intensity counters
        cum_grad_top = torch.sum(torch.abs(aug[0,:int(h/2),int(w/2)]))
        cum_grad_bot = torch.sum(torch.abs(aug[0,int(h/2):,int(w/2)]))
        cum_top = 0
        cum_bot = 0
        cum_sum_top = 0
        cum_sum_bot = 0

        #convert into hls color space to detect lightness easier
        hls_img = cv2.cvtColor(img.permute(1,2,0).numpy(),cv2.COLOR_RGB2HLS)
        hls_img = torch.from_numpy(hls_img)

        #index at which the mask should mark as 0 -> from i to center of image
        mask_ind_top = 0
        mask_ind_bot = h

        for i in range(int(h/2)):

            cum_top += torch.abs(aug[0,i,int(w/2)]) / cum_grad_top
            cum_bot += torch.abs(aug[0,h-1-i,int(w/2)]) / cum_grad_bot

            if (cum_top < 0.3): mask_ind_top = i
            if (cum_bot < 0.3): mask_ind_bot = h-i

            #skip calculations where gradient isnt large enough to notice
            if((cum_top or cum_bot) < 0.1):
                continue

            #cumulated level of deformation (to center) * the intensity param.
            #cum_sum_top += cum_top * self.grad_intensity
            #cum_sum_bot += cum_bot * self.grad_intensity

            #cumulated level of deformation (to center) * the intensity param. use this? top solution kinda shitty for squeeze
            cum_sum_top += torch.abs(aug[0,i,int(w/2)]) / cum_grad_top * self.grad_intensity
            cum_sum_bot += torch.abs(aug[0,h-1-i,int(w/2)]) / cum_grad_bot * self.grad_intensity

            color_threshold_top = (hls_img[i,:,1]<self.color_thr).unsqueeze(0).repeat(3,1)
            color_threshold_bot = (hls_img[h-1-i,:,1]<self.color_thr).unsqueeze(0).repeat(3,1)

            if self.mode == "squeeze": #ca. grad_intensity=1.0,color_thr=0.8

                #saturation
                img[:,i,:] = F.adjust_saturation(img[:,i,:].unsqueeze(1),1+cum_sum_top).squeeze()
                img[:,h-1-i,:] = F.adjust_saturation(img[:,h-1-i,:].unsqueeze(1),1+cum_sum_bot).squeeze()

                #Brightness
                brightnessTop = (1-cum_sum_top*0.3) if (1-cum_sum_top*0.3) > 0.3 else 0.3
                brightnessBot = (1-cum_sum_bot*0.3) if (1-cum_sum_bot*0.3) > 0.3 else 0.3
                img[:,i,:][color_threshold_top] = F.adjust_brightness(img[:,i,:][color_threshold_top].unsqueeze(1),brightnessTop).squeeze()
                img[:,h-1-i,:][color_threshold_bot] = F.adjust_brightness(img[:,h-1-i,:][color_threshold_bot].unsqueeze(1),brightnessBot).squeeze()

            elif self.mode == "stretch": #ca. grad_intensity=2.0,color_thr=0.95

                #saturation
                intensityTop = (1-cum_sum_top*0.3) if (1-cum_sum_top*0.3) > 0.3 else 0.3
                intensityBot = (1-cum_sum_bot*0.3) if (1-cum_sum_bot*0.3) > 0.3 else 0.3
                img[:,i,:] = F.adjust_saturation(img[:,i,:].unsqueeze(1),intensityTop).squeeze()
                img[:,h-1-i,:] = F.adjust_saturation(img[:,h-1-i,:].unsqueeze(1),intensityBot).squeeze()

                #Brightness
                img[:,i,:][color_threshold_top] = F.adjust_brightness(img[:,i,:][color_threshold_top].unsqueeze(1),1+cum_sum_top).squeeze()
                img[:,h-1-i,:][color_threshold_bot] = F.adjust_brightness(img[:,h-1-i,:][color_threshold_bot].unsqueeze(1),1+cum_sum_bot).squeeze()
        
        # max brightness - todo
        if self.mode == "streeetch":
            hls_img = cv2.cvtColor(img.permute(1,2,0).numpy(),cv2.COLOR_RGB2HLS)
            hls_img = torch.from_numpy(hls_img)
            #hls_img[hls_img[:,:,0] > 0.8] = 0.8
            temp = hls_img[:,:,1]
            temp[temp[:,:]>0.9] = 0.8
            hls_img[:,:,1] =  temp
            img = cv2.cvtColor(hls_img.numpy(),cv2.COLOR_HLS2RGB)
            img = torch.from_numpy(img)
            img = img.permute(2,0,1)
            
            
            
        #img = F.adjust_brightness(img,1.5)
        sample["image"] = img
        aug = aug[0,:,:]
        # ----------------- meh, probably change mask somehow --------------------
        sample["ood_mask"][np.abs(aug) > 5] = 0
        sample["ood_mask"][mask_ind_top:mask_ind_bot,:] = 0
        #sample["ood_mask"] = mask

        return sample
