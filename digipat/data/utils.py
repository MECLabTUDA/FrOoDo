import torch
import numpy as np


def augmentation_collate(d):
    return (
        torch.stack([i for i, _, _ in d]),
        torch.stack([m for _, m, _ in d]),
        [meta for _, _, meta in d],
    )


def whole_binning(value, range, num_bins=None, ignore_true_bin=False, default=0):
    return get_standard_bin(
        value, create_bins(range, num_bins), ignore_true_bin, default
    )


def create_bins(range, num_bins):
    if num_bins == None:
        return None
    min_value, max_value = range
    return np.linspace(min_value + max_value / num_bins, max_value, num_bins)


def get_standard_bin(value, bins, ignore_true_bin=False, default=0):
    if value == default:
        return -1

    # if bins are set to None this means the true bin is not needed and only the information whether the data is IN or OOD is reqiered
    if ignore_true_bin:
        return True

    return np.digitize([value], bins, right=True)[0]
