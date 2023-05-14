import os
import random
from froodo import SampleDataset
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from froodo import Sample

import numpy as np
import matplotlib.pyplot as plt
from froodo.data.metadata.types import SampleMetadataCommonTypes
from froodo.ood.augmentations.endoscopy.blood import BloodAugmentation
from froodo.ood.augmentations.endoscopy.corn import CornAugmentation
from froodo.ood.augmentations.endoscopy.pill import PillAugmentation
from froodo.ood.augmentations.endoscopy.random_hue_shifts import RandomHueShiftAugmentation
from froodo.ood.augmentations.endoscopy.random_value_shifts import RandomValueShiftAugmentation

from froodo.ood.augmentations.endoscopy.vignette import Vignette
from froodo.ood.augmentations.endoscopy.coins import CoinAugmentation

from froodo.ood.augmentations.endoscopy.kidney_bean import KidneyBeanAugmentation
from froodo.ood.augmentations.endoscopy.chili import ChiliAugmentation
from froodo.ood.augmentations.endoscopy.green_onion import GreenOnionAugmentation
from froodo.ood.augmentations.endoscopy.pumpkin_seed import PumpkinSeedAugmentation
from froodo.ood.augmentations.endoscopy.melon_seed import MelonSeedAugmentation
from froodo.ood.augmentations.endoscopy.oats import OatsAugmentation
from froodo.ood.augmentations.endoscopy.coin_v2 import CoinV2Augmentation
from froodo.ood.augmentations.endoscopy.pill_v2 import PillV2Augmentation


from froodo.data.datasets.examples.endoscopy.ksavir import KsavirDataset

from torch.utils.data import DataLoader

"""
class KsavirDataset(Dataset,SampleDataset):
    def __init__(self, root_path: str) -> None:
        super().__init__()
        self.root_path = root_path
        #self.images = glob.glob(root_path + "/*/")
        self.images = []
        for folder in os.listdir(root_path):
            self.images += [folder + "/" + file for file in os.listdir(root_path +"/"+folder)]
        self.convert_to_tensor = transforms.PILToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> torch.Tensor:
        img = Image.open(self.root_path + "/" + self.images[index])
        image = self.convert_to_tensor(img) / 255
        return Sample(image)
"""


ksavir_dataset = KsavirDataset("../FrOoDo/Datasets/kvasir-dataset-v2", True)
assert len(ksavir_dataset)==8000
sample = random.randint(0, len(ksavir_dataset))
#sample = 4875
#sample = 5341
#sample = 3736
#sample = 1602
#sample = 5731
#sample = 3903
print("sample Num: ", sample)
sample = ksavir_dataset[sample]
#sample = Nothing()(sample)

"""
plt.imshow(sample.image.permute(1,2,0))
plt.imshow(Vignette()(sample).image.permute(1,2,0))
plt.imshow(CoinAugmentation()(sample).image.permute(1,2,0))
plt.show()
"""


#sample = BloodAugmentation()(sample)
#sample = RandomValueShiftAugmentation()(sample)
#sample = CoinAugmentation()(sample)
#sample = CornAugmentation()(sample)
#sample = PillAugmentation()(sample)
#sample = RandomHueShiftAugmentation()(sample)
#sample = KidneyBeanAugmentation()(sample)
#sample = GreenOnionAugmentation()(sample)
#sample = ChiliAugmentation()(sample)
#sample = PumpkinSeedAugmentation()(sample)
#sample = MelonSeedAugmentation()(sample)
sample = OatsAugmentation()(sample)
sample = CoinV2Augmentation()(sample)
sample = PillV2Augmentation()(sample)




sample = Vignette(0.2)(sample)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(sample.image.permute(1,2,0))
fig.add_subplot(1, 2, 2)
plt.imshow(sample['ood_mask'])



#sample.plot()
print(sample['ood_mask'])
print(torch.bincount(sample['ood_mask'].flatten().long()))
print(sample["metadata"][SampleMetadataCommonTypes.OOD_SEVERITY.name])


plt.show()