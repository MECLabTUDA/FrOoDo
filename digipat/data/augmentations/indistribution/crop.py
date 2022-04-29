from .. import INAugmantation


class InCrop(INAugmantation):
    def __init__(self, crop_size=(300, 300)) -> None:
        super().__init__()
        self.crop_size = crop_size

    def __call__(self, img, mask, metadata):
        return super().__call__(img, mask, metadata)
