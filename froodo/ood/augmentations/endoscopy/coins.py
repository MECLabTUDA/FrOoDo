
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

class CoinAugmentation(OODAugmentation):
    def __init__(self) -> None:
        super().__init__()

    def _apply_sampling(self):
        return super()._apply_sampling()
        

    def _clamp(self, value, miN, maX):
        return max(min(value, maX), miN)

    
    def _augment(self, sample: Sample) -> Sample:
        path = f"froodo/ood/augmentations/endoscopy/artifacts/imgs/coins/{random.choice(listdir('froodo/ood/augmentations/endoscopy/artifacts/imgs/coins'))}"
        
        img = sample.image.permute(1, 2, 0)     #CHW -> HWC
        ood_mask = sample['ood_mask']

        H, W, _ = img.shape

        overlay = imageio.imread(path) / 255.0
        overlay = cv.resize(overlay, (0, 0), fx=0.125, fy=0.125)
        overlay_alpha_channel = torch.from_numpy(overlay[:, :, 3])


        ################## compute location for insertion
        H_overlay, W_overlay, _ = overlay.shape
        insert_height = int(np.random.normal(H/2, H/4))
        insert_width = int(np.random.normal(W/2, W/4))

        insert_height -= H_overlay//2
        insert_width -= W_overlay//2

        ## these would be possible, but does not make that much sense
        #insert_height = self._clamp(insert_height, -H_overlay, H)
        #insert_width = self._clamp(insert_width, -W_overlay, W)

        insert_height = self._clamp(insert_height, 0, H-H_overlay)
        insert_width = self._clamp(insert_width, 0, W-W_overlay)


        ################# set brigthness
        background_pixel = img[insert_height + H_overlay//2, insert_width + W_overlay//2]
        _, _, background_brightness = colorsys.rgb_to_hsv(background_pixel[0], background_pixel[1], background_pixel[2])

        foreground_pixel = overlay[H_overlay//2,W_overlay//2]
        _, _, foreground_brightness = colorsys.rgb_to_hsv(foreground_pixel[0], foreground_pixel[1], foreground_pixel[2])


        overlay = np.float32(overlay)

        align_brightness = True
        if align_brightness:
            hsv = cv.cvtColor(overlay[:,:,:-1], cv.COLOR_RGB2HSV)
            #hsv[:,:,2] = background_brightness                                             # constant
            hsv[:,:,2] = hsv[:,:,2] * background_brightness.item() /  foreground_brightness # relative to foreground pixel 
            overlay = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

            overlay = np.dstack((overlay,overlay_alpha_channel))


        ########## execute overlaying
        from_y_art = max(-insert_height, 0)
        from_x_art = max(-insert_width, 0)
        from_y = -min(-insert_height, 0)
        from_x = -min(-insert_width, 0)
        until_y = min(H_overlay - from_y_art + from_y, H)
        until_x = min(W_overlay - from_x_art + from_x, W)
        until_y_art = from_y_art + until_y - from_y
        until_x_art = from_x_art + until_x - from_x

        one_minus_alpha = (1 - overlay_alpha_channel[from_y_art:until_y_art, from_x_art:until_x_art, np.newaxis])
        
        overlayed = (
            overlay_alpha_channel[from_y_art:until_y_art, from_x_art:until_x_art, np.newaxis]
            * overlay[from_y_art:until_y_art, from_x_art:until_x_art, :3]
            + one_minus_alpha
            * img[from_y:until_y, from_x:until_x, :]
        )
        img[from_y:until_y, from_x:until_x, :] = overlayed

        ood_indices = torch.from_numpy(
            overlay[from_y_art:until_y_art, from_x_art:until_x_art, 3] > 0.3
        )
        ood_mask[from_y:until_y, from_x:until_x][ood_indices] = 0

        sample["image"] = img.permute(2,0,1)    #HWC -> CHW
        sample["ood_mask"] = ood_mask

        return sample