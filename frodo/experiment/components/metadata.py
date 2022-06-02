from collections import defaultdict

from ...utils import dict_default


class ComponentMetadata:
    def __init__(self, data_dict=None) -> None:
        if data_dict == None:
            self.data = defaultdict(dict_default)
        else:
            self.data = defaultdict(dict_default, data_dict)

    def __getitem__(self, arg):
        return self.data[arg]

    def __setitem__(self, idx, value):
        self.data[idx] = value
