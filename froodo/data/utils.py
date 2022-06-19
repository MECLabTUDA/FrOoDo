from .samples import Batch


def sample_collate(d):
    return Batch.init_from_samples(d)
