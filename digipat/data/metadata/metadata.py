from collections import defaultdict

from ..datatypes import DistributionSampleType
from ...utils import dict_default


class SampleMetadata:
    def __init__(
        self, type=DistributionSampleType.IN_DATA, data_dict: defaultdict = None
    ) -> None:
        if data_dict == None:
            self.data = defaultdict(dict_default)
        else:
            assert isinstance(data_dict, defaultdict)
            self.data = data_dict
        self.type = type

    @property
    def type(self):
        return self._type

    @property
    def data(self):
        return self._data

    @type.setter
    def type(self, sample_type: DistributionSampleType):
        assert isinstance(
            sample_type, DistributionSampleType
        ), f"Metadata Type needs to be a DistributionSampleType ({type(sample_type)} is invalid)"
        self._type = sample_type
        self._data["type"] = sample_type

    @data.setter
    def data(self, data_dict: defaultdict):
        assert isinstance(
            data_dict, defaultdict
        ), "Metadata Data needs to be a dictionary"
        self._data = data_dict

    def __getitem__(self, arg):
        if arg == "type":
            return self.type
        return self.data[arg]

    def __setitem__(self, idx, value):
        if idx == "type":
            self.type = value
            return
        self.data[idx] = value

    def __delitem__(self, item: str) -> None:
        if item == "type":
            print("Type cant be deleted")
        else:
            self._data.__delattr__(item)

    def __repr__(self) -> str:
        return f"{self.type} - {self.data.items()}"
