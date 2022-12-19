from torch.utils.data import Dataset
import torchvision as tv
from froodo import *


class XrayDatasetAdapter(Dataset, SampleDataset):
    def __init__(self, path, **kwargs) -> None:
        transform = tv.transforms.Compose([tv.transforms.Resize(255),
                                           tv.transforms.CenterCrop(224),
                                           tv.transforms.ToTensor()])
        self.dataset = tv.datasets.ImageFolder(path, transform=transform)
        self.metadata_args = kwargs

    def _add_metadata_args(self, sample: Sample) -> Sample:
        sample.metadata.data.update(self.metadata_args)
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Sample:
        sample = self.dataset[index]
        return Sample(sample[0], {"class_label": torch.Tensor([[sample[1]]])})
