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
        if show:
            f, axes = plt.subplots(2, n, figsize=(n * 4, 8))
            for i, (img, mask) in enumerate(data):
                axes[0, i].imshow(img.numpy().transpose((1, 2, 0)))
                axes[1, i].imshow(mask.numpy(), vmin=0, vmax=5)
                axes[0, i].axis("off")
                axes[1, i].axis("off")
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
