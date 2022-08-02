from torchvision import transforms
import torch
from medmnist import PathMNIST, DermaMNIST, BloodMNIST, RetinaMNIST, TissueMNIST
from torchvision.transforms import Compose, ToTensor

from typing import Union

from ...adapter import DatasetAdapter
from ....samples import Sample


class MedMNISTAdapter(DatasetAdapter):
    def __init__(self, dataset_class, split="test", transform=None, **kwargs) -> None:
        super().__init__(
            dataset_class(
                split=split,
                as_rgb=True,
                transform=ToTensor() if transform == None else transform,
            ),
            dataset_class.__name__,
            **kwargs
        )

    def __getitem__(self, index) -> Sample:
        image, label = self.dataset[index]
        return Sample(image, {"class_label": torch.tensor(label[0])})


class PathMNIST_Adapted(MedMNISTAdapter):
    def __init__(self, split="test", transform=None, **kwargs) -> None:
        super().__init__(PathMNIST, split, transform, **kwargs)


class DermaMNIST_Adapted(MedMNISTAdapter):
    def __init__(self, split="test", transform=None, **kwargs) -> None:
        super().__init__(DermaMNIST, split, transform, **kwargs)


class BloodMNIST_Adapted(MedMNISTAdapter):
    def __init__(self, split="test", transform=None, **kwargs) -> None:
        super().__init__(BloodMNIST, split, transform, **kwargs)


class RetinaMNIST_Adapted(MedMNISTAdapter):
    def __init__(self, split="test", transform=None, **kwargs) -> None:
        super().__init__(RetinaMNIST, split, transform, **kwargs)


class TissueMNIST_Adapted(MedMNISTAdapter):
    def __init__(self, split="test", transform=None, **kwargs) -> None:
        super().__init__(TissueMNIST, split, transform, **kwargs)
