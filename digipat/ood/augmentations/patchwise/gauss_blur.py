from torchvision.transforms import GaussianBlur

from ...augmentations import OODAugmantation


class GaussianBlurAugmentation(OODAugmantation):
    def __init__(self, kernel_size=(19, 19), sigma=(1.3, 2.5), prob=1) -> None:
        super().__init__()
        self.augmentation = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.prob = prob

    def __call__(self, img, mask):
        return super().do_random(img, mask, augmentation=self.augmentation)
