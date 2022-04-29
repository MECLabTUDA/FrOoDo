import numpy as np

import functools


def binning_fn(param_range):
    return functools.partial(whole_binning, range=param_range)


def whole_binning(value, num_bins=None, ignore_true_bin=False, range=None, default=0):
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
