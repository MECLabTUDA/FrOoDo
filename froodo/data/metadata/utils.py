from .metadata import SampleMetadata
from .batch import SampleMetdadataBatch


def init_sample_metadata(metadata=None):
    if metadata == None:
        return SampleMetadata()
    else:
        assert isinstance(metadata, SampleMetadata) or isinstance(
            metadata, SampleMetdadataBatch
        ), "Metadata needs to be class Metadata or MetdataBatch"
        return metadata
