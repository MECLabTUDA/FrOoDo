from copy import deepcopy

import matplotlib.pyplot as plt
import torch

from examples.XrayDatasetAdapter import XrayDatasetAdapter
from froodo import TubesAugmentation, Nothing, GaussianNoiseAugmentation, SampleMetadataCommonTypes, \
    MotionBlurAugmentation
from froodo.ood.augmentations.xray.artifacts.coin import CoinAugmentation
from froodo.ood.augmentations.xray.artifacts.foreign_bodies import ForeignBodiesAugmentation

xray_dataset = XrayDatasetAdapter('~/Downloads/chest_xray/test')
#sampleIndex = random.randint(0, len(xray_dataset))
sampleIndex = 100
sample = xray_dataset[sampleIndex]

#single = GaussianNoiseAugmentation(sigma=0.01)(sample)
#print(single["metadata"][SampleMetadataCommonTypes.OOD_SEVERITY.name])
#sample = xray_dataset[sampleIndex]

augmentations = [
    #[Nothing(), Nothing()],
    [TubesAugmentation(amount=1), TubesAugmentation(amount=6)],
    [ForeignBodiesAugmentation(amount=1), ForeignBodiesAugmentation(amount=6)],
    [CoinAugmentation(amount=1), CoinAugmentation(amount=2)],
    [GaussianNoiseAugmentation(sigma=0.001), GaussianNoiseAugmentation(sigma=0.01)],
    [MotionBlurAugmentation(motion=5), MotionBlurAugmentation(motion=20)]
]

titles = ["Tubes", "Foreign Bodies", "Coin", "Noise", "Motion"]

f, ax = plt.subplots(4, len(augmentations), figsize=(5 * len(augmentations), 20))

ax[0, 0].set_ylabel("Image", fontsize=35)
ax[1, 0].set_ylabel("OOD Mask", fontsize=35)
ax[2, 0].set_ylabel("Image", fontsize=35)
ax[3, 0].set_ylabel("OOD Mask", fontsize=35)

for i in range(len(augmentations)):
    for j in range(len(augmentations[i])):
        if j == 0:
            ax[0, i].set_title(titles[i], fontsize=35)
        s = deepcopy(sample)
        s = augmentations[i][j](s)
        ax[2*j, i].imshow(s.image.permute(1, 2, 0))
        ax[2*j+1, i].imshow(s["ood_mask"], vmin=0, vmax=1)
        ax[2*j, i].set_yticks([])
        ax[2*j, i].set_xticks([])
        ax[2*j+1, i].set_yticks([])
        ax[2*j+1, i].set_xticks([])
plt.show()

