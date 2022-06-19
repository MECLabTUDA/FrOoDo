import torch

from collections import defaultdict
import matplotlib.pyplot as plt

from ...utils import dict_default
from ..metadata import SampleMetadata, init_sample_metadata


class Sample:
    def __init__(self, image, label_dict: dict = None, metadata=None) -> None:
        self.image = image
        if label_dict == None:
            self.label_dict = defaultdict(dict_default)
        else:
            self.label_dict = defaultdict(dict_default, label_dict)
        self.metadata = init_sample_metadata(metadata)

    def add_labels_dict(self, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(v, torch.Tensor)
            self.label_dict[k] = v

    def add_segmentation_mask(self, mask: torch.Tensor):
        self.label_dict["segmentation_mask"] = mask

    def add_ood_mask(self, mask: torch.Tensor):
        self.label_dict["ood_mask"] = mask

    def add_class_vector(self, class_vector: torch.Tensor):
        self.label_dict["class"] = class_vector

    def get(self):
        return self.image, self.label_dict, self.metadata

    def __getitem__(self, arg):
        if arg == "image":
            return self.image
        elif arg == "metadata":
            return self.metadata
        return self.label_dict[arg]

    def __setitem__(self, idx, value):
        if idx == "image":
            assert isinstance(value, torch.Tensor)
            self.image = value
            return
        if idx == "metadata":
            assert isinstance(value, SampleMetadata)
            self.metadata = value
            return
        assert isinstance(value, torch.Tensor)
        self.label_dict[idx] = value

    def plot(self, print_meta=True):
        f, ax = plt.subplots(
            1, 1 + len(self.label_dict), figsize=(4 * (1 + len(self.label_dict)), 4)
        )
        ax[0].imshow(self.image.permute(1, 2, 0))
        ax[0].set_title(f"image {list(self.image.size())}")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        for i, (k, v) in enumerate(self.label_dict.items()):
            ax[i + 1].imshow(v)
            ax[i + 1].set_title(f"{k} {list(v.size())}")
            ax[i + 1].set_xticks([])
            ax[i + 1].set_yticks([])
        plt.show()
        if print_meta:
            print(self.metadata)
