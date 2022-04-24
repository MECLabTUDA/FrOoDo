import torch

from .utils import *


class SeverityMeasurement:
    def __init__(self, binning_fn, name):
        self.binning_fn = binning_fn
        self.name = name

    def calculate_measurement(self, image, mask, params):
        raise NotImplementedError("Please Implement this method")

    def get_bin(self, num_bins=None, ignore_true_bin=False):
        return self.binning_fn(self.measurement, num_bins, ignore_true_bin)

    def __repr__(self) -> str:
        return f"{self.name} [{self.range[0]}-{self.range[1]}]: {self.measurement}"


class PixelPercentageSeverityMeasurement(SeverityMeasurement):
    def __init__(self):
        super().__init__(
            lambda value, num_bins=None, ignore_true_bin=False: whole_binning(
                value=value,
                range=(0, 1),
                num_bins=num_bins,
                ignore_true_bin=ignore_true_bin,
            ),
            "PixelPercentage",
        )
        self.range = (0, 1)

    def calculate_measurement(self, image, mask, params):
        self.measurement = (
            torch.sum(mask == 0) / (mask.shape[-1] * mask.shape[-2])
        ).item()
        return self.measurement


class ParameterSeverityMeasurement(SeverityMeasurement):
    def __init__(self, param, param_range, custom_binning_fn=None):
        super().__init__(
            lambda value, num_bins=None, ignore_true_bin=False: whole_binning(
                value=value,
                range=param_range,
                num_bins=num_bins,
                ignore_true_bin=ignore_true_bin,
            )
            if custom_binning_fn == None
            else custom_binning_fn,
            f"Parameter {param}",
        )
        self.param = param
        self.range = param_range

    def calculate_measurement(self, image, mask, params):
        self.measurement = params[self.param]
        return self.measurement
