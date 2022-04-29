from .metadata import SampleMetadata


def init_sample_metadata(metadata=None):
    if metadata == None:
        return SampleMetadata()
    else:
        assert isinstance(
            metadata, SampleMetadata
        ), "Metadata needs to be class Metadata"
        return metadata
