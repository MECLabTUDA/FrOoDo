## Dataset adaptation

To use own datasets one needs to adapts its dataset to this sample standard. This can be done by a dataset adapter which is essentially an own dataset with only the purpose to translate the dataset output from your dataset to the SampleDataset.

```python
# Adapter
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
```

For the most usecases there are already Adapters that one can use. 

```python
dataset = my dataset

# use this if your dataset already returns a tuple of image, a label dictionary and optionally metadata
adapter = ImageLabelMetaAdapter(dataset)

# use this dapter if your dataset returns a tuple of tensors
adapter = PositionalAdapter(dataset, remapping = {"image": 0, "segmentation_mask": 1})

# use this adapter if your dataset returns a dictionary
adapter = DatasetWithAllInOneDictAdapter(dataset, remapping= {
    "image": "img",
    "segmentation_mask": "mask"
})

# use utility function to create mapping
def create_mapping_dict(
    image, ood_mask=None, segmentation_mask=None, class_vector=None, **kwargs
)
```

```python
# test if adaptation worked
dataset.sample()
```