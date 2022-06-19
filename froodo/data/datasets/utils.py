import tifffile as tiff
import zarr
import numpy as np
import torch

import random
from os.path import join


def read_tif_region(file, from_x=None, to_x=None, from_y=None, to_y=None):
    store = tiff.imread(file, aszarr=True)
    out_labels_slide = zarr.open(store, mode="r")

    if from_x == None:
        return out_labels_slide
    return out_labels_slide[from_y:to_y, from_x:to_x]


def random_crop(img, mask, height, width):
    x = random.randint(0, img.shape[2] - width)
    y = random.randint(0, img.shape[1] - height)
    img = img[:, y : y + height, x : x + width]
    mask = mask[y : y + height, x : x + width]
    return (img, mask)


def get_valid_regions_of_image(
    folder,
    image,
    size,
    overlap,
    ignore_classes,
    non_ignore_threshold=0.9,
):

    img = read_tif_region(join(folder, image))

    ov_y = size[0] - int(size[0] * overlap)
    ov_x = size[1] - int(size[1] * overlap)

    num_pixel = size[0] * size[1]
    img_size = img.shape

    tiles = []
    num_valid = 0
    num_invalid = 0

    y = size[0]
    x = size[1]

    while y <= img_size[0]:

        x = size[1]
        while x < img_size[1]:
            mask = img[y - size[0] : y, x - size[1] : x]
            perc = np.sum(np.in1d(mask, [ignore_classes])) / num_pixel

            if mask.shape[0] != size[0] or mask.shape[1] != size[1]:
                print("falsch")

            if perc <= non_ignore_threshold:
                tiles.append([x - size[1], x, y - size[0], y])
                num_valid += 1
            else:
                num_invalid += 1

            x += ov_x

        y += ov_y

    return tiles, [num_valid, num_invalid]


def to_tensor(array, mask=False):
    if not mask:
        return torch.from_numpy(array).permute(2, 0, 1)
    else:
        return torch.from_numpy(array)


def apply_mask_changes(
    mask,
    ignore_classes,
    ood_classes,
    ignore_index,
    remain_classes,
    map_classes,
    mode=["mask"],
):
    masks = {}
    if "mask" in mode:
        _mask = mask.copy()
        for f, t in map_classes:
            _mask[_mask == f] = t
        _mask[np.in1d(_mask, ignore_classes + ood_classes).reshape(_mask.shape)] = (
            ignore_index + 1
        )
        _mask = _mask - 1
        masks["segmentation_mask"] = to_tensor(_mask.astype(np.longlong), True)

    if "ood" in mode or "full_ood" in mode:
        _mask = mask.copy()
        for f, t in map_classes:
            _mask[_mask == f] = t
        remain_indices = np.in1d(_mask, remain_classes).reshape(_mask.shape)
        ood_indices = np.in1d(_mask, ood_classes).reshape(_mask.shape)
        ignore_indices = np.in1d(_mask, ignore_classes).reshape(_mask.shape)
        _mask[remain_indices] = 1
        _mask[ood_indices] = 0 if "ood" in mode else ignore_index
        _mask[ignore_indices] = ignore_index
        masks["ood_mask"] = to_tensor(_mask.astype(np.longlong), True)
    return masks
