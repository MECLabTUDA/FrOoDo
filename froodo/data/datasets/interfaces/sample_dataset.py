import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from typing import Union

from ...utils import sample_collate
from ...datatypes import DistributionSampleType


class SampleDataset:

    name: Union[str, None] = None

    def find_base_name(self, default=None):
        if self.name != None:
            return self.name

        if hasattr(self, "dataset") and isinstance(self.dataset, SampleDataset):
            return self.dataset.find_base_name(default)

        return default

    def get_dataloader(
        self,
        batch_size: int = 16,
        num_workers: Union[int, None] = 10,
        shuffle: bool = False,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        **dataloader_kwargs,
    ) -> DataLoader:
        return DataLoader(
            self,
            collate_fn=sample_collate,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            persistent_workers=persistent_workers,
            **dataloader_kwargs,
        )

    def sample(self, n=8, show=True, random=True):
        if random:
            ids = np.random.choice(
                range(len(self)),
                n,
            )
        else:
            ids = [i + 50 for i in range(n)]
        data = [self[i] for i in ids]

        keys = data[0].label_dict.keys()
        mask_overlap = set(keys).intersection(
            {
                "segmentation_mask",
                "ood_mask",
            }
        )
        n_masks = len(mask_overlap)

        if show:
            f, axes = plt.subplots(1 + n_masks, n, figsize=(n * 4, (n_masks + 1) * 4))

            # expand dims if no mask is found
            if n_masks == 0:
                axes = np.expand_dims(axes, axis=0)

            axes[0, 0].set_ylabel("image", rotation=90, size=20)
            axes[0, 0].yaxis.set_label_coords(-0.1, 0.5)
            for i, sample in enumerate(data):
                axes[0, i].imshow(sample["image"].numpy().transpose((1, 2, 0)))
                axes[0, i].set_yticks([])
                axes[0, i].set_xticks([])
                j = 0
                axes[0, i].set_title(
                    sample.metadata.type.name,
                    fontsize=25,
                    c="red"
                    if sample.metadata.type is DistributionSampleType.OOD_DATA
                    else "black",
                )
                for (k, v) in sample.label_dict.items():
                    if k in mask_overlap:
                        axes[j + 1, i].imshow(v.numpy(), vmin=0, vmax=5)
                        axes[j + 1, i].set_yticks([])
                        axes[j + 1, i].set_xticks([])
                        if i == 0:
                            axes[j + 1, i].set_ylabel(k, rotation=90, size=20)
                            axes[j + 1, i].yaxis.set_label_coords(-0.1, 0.5)
                        j += 1
                    elif k == "class_label":
                        axes[0, i].set_title(
                            f"Class {int(v.item())}",
                            fontsize=25,
                            c="red"
                            if sample.metadata.type is DistributionSampleType.OOD_DATA
                            else "black",
                        )

            plt.show()
