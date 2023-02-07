
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

from froodo.ood.severity.severity import PixelPercentageSeverityMeasurement

class ChiliAugmentation(OODAugmentation):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 0.25
        self.severity_class = PixelPercentageSeverityMeasurement()

    def _apply_sampling(self):
        return super()._apply_sampling()
        

    def _clamp(self, value, miN, maX):
        return max(min(value, maX), miN)

    
    def _augment(self, sample: Sample) -> Sample:
        # Settings
        align_brightness = True
        sample_uniformly = True
        random_rotate = True


        path = f"froodo/ood/augmentations/endoscopy/artifacts/imgs/chili/{random.choice(listdir('froodo/ood/augmentations/endoscopy/artifacts/imgs/chili'))}"
        
        img = sample.image.permute(1, 2, 0)     #CHW -> HWC

        img = img.numpy()
        img *= 255
        img = img.astype(np.uint8)

        img_denoised = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 7)
        img_denoised = img_denoised.tolist()
        img_denoised = np.array(img_denoised)

        noise = img - img_denoised

        noise = torch.from_numpy(noise.astype(np.float32))
        noise /= 255

        img = torch.from_numpy(img_denoised.astype(np.float32))
        img /= 255

        ood_mask = sample['ood_mask']

        H, W, _ = img.shape

        overlay = imageio.imread(path) / 255.0
        overlay = cv.resize(overlay, (0, 0), fx=self.scale, fy=self.scale)

        if random_rotate:
            rotation_angle = np.random.randint(0, high=360)
            overlay = scipy.ndimage.rotate(overlay, rotation_angle)





        overlay_alpha_channel = torch.from_numpy(overlay[:, :, 3])


        ################## compute location for insertion
        H_overlay, W_overlay, _ = overlay.shape
        if sample_uniformly:
            insert_height = np.random.randint(0, high=H-H_overlay +1)
            insert_width =  np.random.randint(0, high=W-W_overlay +1)
        else:
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

        if align_brightness:
            hsv = cv.cvtColor(overlay[:,:,:-1], cv.COLOR_RGB2HSV)
            #hsv[:,:,2] = background_brightness                                             # constant
            hsv[:,:,2] = hsv[:,:,2] * background_brightness.item() /  foreground_brightness # relative to foreground pixel 
            overlay = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

            overlay = np.clip(overlay,a_min=0,a_max=1)
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

        img += noise

        
        sample["image"] = img.permute(2,0,1)    #HWC -> CHW
        sample["ood_mask"] = ood_mask

        return sample