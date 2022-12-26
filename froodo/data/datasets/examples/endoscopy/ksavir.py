import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from froodo import Sample
from froodo import SampleDataset

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