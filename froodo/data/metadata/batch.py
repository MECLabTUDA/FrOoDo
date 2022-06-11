from typing import List

from .metadata import SampleMetadata


class SampleMetdadataBatch:
    def __init__(self, batch: List[SampleMetadata]) -> None:
        self.batch = batch
        # only the created keys are saved here
        self.created = []

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, index):
        return self.batch[index]

    def __getitem__(self, arg):
        return [m[arg] for m in self.batch]

    def __setitem__(self, idx, value):
        for m in self.batch:
            m[idx] = value

    def init_as_dict(self, key):
        for m in self.batch:
            if key not in m.data:
                m[key] = {}

    def append_to_keyed_dict(self, key, dict_key, items):
        if key not in self.created:
            self.init_as_dict(key)
        if len(self) == len(items):
            for i, m in enumerate(self.batch):
                m[key][dict_key] = items[i]
