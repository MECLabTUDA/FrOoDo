from copy import deepcopy
import matplotlib.pyplot as plt
from froodo import PneumoniaDataSetAdapter, TubesAugmentation, ForeignBodiesAugmentation, CoinAugmentation, NailsAugmentation, GaussianNoiseAugmentation, MotionBlurAugmentation

dataset_adapter = PneumoniaDataSetAdapter('~/Downloads/chest_xray/', split='test')
sampleIndex = 100
sample = dataset_adapter[sampleIndex]

#single = GaussianNoiseAugmentation(sigma=0.01)(sample)
#print(single["metadata"][SampleMetadataCommonTypes.OOD_SEVERITY.name])
#sample = xray_dataset[sampleIndex]

augmentations = [
    [TubesAugmentation(amount=1, keep_ignored=False), TubesAugmentation(amount=6, keep_ignored=False)],
    [ForeignBodiesAugmentation(amount=1, keep_ignored=False), ForeignBodiesAugmentation(amount=6, keep_ignored=False)],
    [CoinAugmentation(amount=1, keep_ignored=False), CoinAugmentation(amount=2, keep_ignored=False)],
    [NailsAugmentation(amount=1, keep_ignored=False), NailsAugmentation(amount=3, keep_ignored=False)],
    [GaussianNoiseAugmentation(sigma=0.001, keep_ignored=False), GaussianNoiseAugmentation(sigma=0.01, keep_ignored=False)],
    [MotionBlurAugmentation(motion=5, keep_ignored=False), MotionBlurAugmentation(motion=20, keep_ignored=False)]
]

titles = ["Tubes", "Foreign Bodies", "Coin", "Nails", "Noise", "Motion"]

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

