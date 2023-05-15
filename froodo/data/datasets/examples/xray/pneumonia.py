from os import path
import torchvision as tv
import torch
from ...adapter.adapter import DatasetAdapter, Sample

"""
Uses the chest xray pneumonia dataset from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""


class PneumoniaDataSetAdapter(DatasetAdapter):
    def __init__(self, dir_path, split="test", **kwargs) -> None:
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(255),
                tv.transforms.CenterCrop(224),
                tv.transforms.ToTensor(),
            ]
        )
        dataset = tv.datasets.ImageFolder(
            path.join(dir_path, split), transform=transform
        )
        super().__init__(dataset, *kwargs)

    def _add_metadata_args(self, sample: Sample) -> Sample:
        sample.metadata.data.update(self.metadata_args)
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Sample:
        sample = self.dataset[index]
        return Sample(sample[0], {"class_label": torch.Tensor([[sample[1]]])})
