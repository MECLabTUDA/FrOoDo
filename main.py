import os
from froodo import SampleDataset
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from froodo import Sample

import numpy as np
import matplotlib.pyplot as plt

from froodo.ood.augmentations.endoscopy.vignette import Vignette
from froodo.ood.augmentations.endoscopy.coins import CoinAugmentation

from froodo.data.datasets.examples.endoscopy.ksavir import KsavirDataset


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


ksavir_dataset = KsavirDataset("G:\FrOoDo\Datasets\kvasir-dataset-v2")
assert len(ksavir_dataset)==8000


sample = ksavir_dataset[900]
#sample = Nothing()(sample)

"""
plt.imshow(sample.image.permute(1,2,0))
plt.imshow(Vignette()(sample).image.permute(1,2,0))
plt.imshow(CoinAugmentation()(sample).image.permute(1,2,0))
plt.show()
"""


sample = CoinAugmentation()(sample)
sample = Vignette()(sample)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(sample.image.permute(1,2,0))
fig.add_subplot(1, 2, 2)
plt.imshow(sample['ood_mask'])
plt.show()



#sample.plot()
print(sample['ood_mask'])
#print(torch.bincount(sample['ood_mask'].flatten().long()))