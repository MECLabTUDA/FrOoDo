import numpy as np

import random


def dict_default():
    return None


def sample_from_intervals(intervals):
    weights = np.array([i[1] - i[0] for i in intervals])
    interval_index = np.random.choice(
        list(range(len(intervals))), p=weights / weights.sum()
    )
    return random.uniform(intervals[interval_index][0], intervals[interval_index][1])
