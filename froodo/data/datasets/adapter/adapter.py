from torch.utils.data import Dataset
import torch

import copy
from typing import Dict

from ...metadata.metadata import SampleMetadata
from ...samples import Sample
from ..interfaces import SampleDataset


class GeneralDatasetAdapter(Dataset, SampleDataset):
    def __init__(self, dataset, remapping: Dict = None, **kwargs):
        if isinstance(dataset, SampleDataset):
            self.adapter = AlreadyASampleAdapter(dataset, **kwargs)
        sample = dataset[0]
        if remapping == None:
            if isinstance(sample, Sample):
                self.adapter = AlreadyASampleAdapter(dataset, **kwargs)
            else:
                assert len(sample) in [
                    2,
                    3,
                ], "If no remapping is given dataset should return a tuple with length of 2 (without metadata) or 3 (with metadata)"
                assert isinstance(
                    sample[0], torch.Tensor
                ), "Image should be of type torch.tensor"
                if len(sample) == 3:
                    assert isinstance(
                        sample[2], SampleMetadata
                    ), "If no remapping is given position 3 of tuple needs to be metadata"
                self.adapter = ImageLabelMetaAdapter(dataset, **kwargs)
        else:
            assert isinstance(remapping, dict), "remapping needs to be a dictionary"
            contains_int = False
            contains_str = False
            for k, v in remapping.items():
                assert isinstance(k, str), "Remapping keys must be strings"
                if isinstance(v, int):
                    assert (
                        not contains_str
                    ), "Remapping values must be of same type (int or sring)"
                    contains_int = True
                if isinstance(v, str):
                    assert (
                        not contains_int
                    ), "Remapping values must be of same type (int or sring)"
                    contains_str = True
                else:
                    assert isinstance(v, int) or isinstance(
                        v, str
                    ), "Remapping values must be of type int or string"

                assert contains_int ^ contains_str

                if contains_int:
                    self.adapter = PositionalAdapter(dataset, remapping, **kwargs)
                elif contains_str:
                    assert isinstance(sample, dict)
                    self.adapter = DatasetWithAllInOneDictAdapter(
                        dataset, remapping, **kwargs
                    )

    def __len__(self):
        return len(self.adapter)

    def __getitem__(self, index) -> Sample:
        return self.adapter.__getitem__(index)


class DatasetAdapter(Dataset, SampleDataset):
    def __init__(self, dataset, **kwargs) -> None:
        self.dataset = dataset
        self.metadata_args = kwargs

    def _add_metadata_args(self, sample: Sample) -> Sample:
        sample.metadata.data.update(self.metadata_args)
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Sample:
        raise NotImplementedError("Please Implement this method")


class AlreadyASampleAdapter(DatasetAdapter):
    def __init__(self, dataset, **kwargs) -> None:
        super().__init__(dataset, **kwargs)

    def __getitem__(self, index) -> Sample:
        sample = self.dataset.__getitem__(index)
        sample = self._add_metadata_args(sample)
        return sample


class ImageLabelMetaAdapter(DatasetAdapter):
    def __init__(self, dataset, **kwargs) -> None:
        super().__init__(dataset, **kwargs)

    def __getitem__(self, index) -> Sample:
        data = self.dataset.__getitem__(index)
        meta = data[2] if len(data) == 3 else None
        # if meta == None:
        #    print("No metadata mapping is giving. System will create blank metdata.")
        sample = Sample(data[0], data[1], meta)
        sample = self._add_metadata_args(sample)
        return sample


class PositionalAdapter(DatasetAdapter):
    def __init__(self, dataset, positional_mapping: Dict[str, int], **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.positional_mapping = positional_mapping
        assert (
            self.positional_mapping.get("image", None) != None
        ), "Please provide an index for 'image'."

    def __getitem__(self, index) -> Sample:
        positional_mapping = copy.copy(self.positional_mapping)
        data_tuple = self.dataset.__getitem__(index)
        meta = (
            data_tuple[positional_mapping]
            if positional_mapping.get("metadata", None) != None
            else None
        )
        sample = Sample(data_tuple[positional_mapping["image"]], metadata=meta)
        positional_mapping.pop("image")
        positional_mapping.pop("metadata", None)
        for k, v in positional_mapping.items():
            sample[k] = data_tuple[v]
        sample = self._add_metadata_args(sample)
        return sample


class DatasetWithAllInOneDictAdapter(DatasetAdapter):
    def __init__(self, dataset, remapping=None, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.label_mapping = copy.copy(remapping)
        if remapping.get("image", None) == None:
            print("No image mapping is giving. Assuming that 'image' is the key.")
            self.image_mapping = "image"
        else:
            self.image_mapping = remapping.get("image")
            self.mask_mapping.pop("image", None)

        if remapping.get("metadata", None) == None:
            print("No metadata mapping is giving. System will create blank metdata.")
            self.metadata_mapping = None
        else:
            self.metadata_mapping = remapping.get("metadata")
            self.mask_mapping.pop("metadata", None)

    def __getitem__(self, index) -> Sample:
        data_dict = self.dataset.__getitem__(index)
        sample = Sample(
            data_dict[self.image_mapping],
            metadata=data_dict[self.metadata_mapping]
            if self.metadata != None
            else None,
        )
        data_dict.pop(self.image_mapping)
        data_dict.pop(self.metadata_mapping, None)
        for k, v in self.label_mapping.items():
            data_dict[k] = data_dict.pop(v)
        sample.add_labels_dict(data_dict)
        sample = self._add_metadata_args(sample)
        return sample
