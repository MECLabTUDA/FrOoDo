import numpy as np
import matplotlib.pyplot as plt
import torch

import random


class UtitlityDataset:
    def sample(self, n=5, show=True, random=True):
        if random:
            ids = np.random.choice(
                range(len(self)),
                n,
            )
        else:
            ids = [i + 50 for i in range(n)]
        data = [self[i] for i in ids]
        n_masks = len(data[0].label_dict)

        if show:
            f, axes = plt.subplots(1 + n_masks, n, figsize=(n * 4, (n_masks + 1) * 4))
            axes[0, 0].set_ylabel("image", rotation=90, size=20)
            axes[0, 0].yaxis.set_label_coords(-0.1, 0.5)
            for i, sample in enumerate(data):
                axes[0, i].imshow(sample["image"].numpy().transpose((1, 2, 0)))
                axes[0, i].set_yticks([])
                axes[0, i].set_xticks([])
                for j, (k, v) in enumerate(sample.label_dict.items()):
                    axes[j + 1, i].imshow(v.numpy(), vmin=0, vmax=5)
                    axes[j + 1, i].set_yticks([])
                    axes[j + 1, i].set_xticks([])
                    if i == 0:
                        axes[j + 1, i].set_ylabel(k, rotation=90, size=20)
                        axes[j + 1, i].yaxis.set_label_coords(-0.1, 0.5)

            plt.show()

    def get_test_image_and_mask(self):
        idx = random.randrange(len(self))
        image, mask = self[idx]
        # print(image.shape, mask.shape)
        image_npy = image.cpu().detach().numpy().transpose((1, 2, 0))
        mask_npy = mask.cpu().detach().numpy()
        img_batch = image.unsqueeze(0)
        if torch.cuda.is_available():
            img_batch = img_batch.cuda()
        return img_batch, image_npy, mask_npy
