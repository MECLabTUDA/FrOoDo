# FrODo - Framework for Out of Distribution Detection

## Requirements

## Samples and Batches and Metadata
To make this a flexible framework it is necesarry to uniform the datatypes used internally. 

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
adapter = PositionalAdapter(dataset, remapping = {"image": 0, "segmantation_mask": 1})

# use this adapter if your dataset returns a dictionary
adapter = DatasetWithAllInOneDictAdapter(dataset, remapping= {
    "image": "img",
    "segmantation_mask": "mask"
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

## OOD Component

```python
# OOD Component

net = SegmentationModel().load()
metrics = [OODAuRoC(group_by="bin", num_bins=50), OODAuRoC()]

component = AugmentationOODEvaluationComponent(
    BCSS_Adapted_Cropped_Resized_Datasets().val,
    SampledOODAugmentation(DarkSpotsAugmentation()),
    net,
    metrics=metrics,
    seed=4321,
)
component()
```

## Add new MetricGroup

```python
class MetricGroup:
    def __init__(self) -> None:
        pass

    def requires(self) -> List[Container]:
        raise NotImplementedError("Please Implement this method")

    def __call__(self, data_container: List[Container], experiment_metadata: dict):
        raise NotImplementedError("Please Implement this method")

    def present(self, **kwargs):
        pass
```

## Add new Method

```python
class OODMethod:
    def __init__(self, hyperparams={}) -> None:
        self.name = "set name"

    def __call__(self, imgs, masks, net):
        raise NotImplementedError("Please Implement this method")

    def modify_net(self, net):
        return net

    def get_params(self, dict=False):
        return "no params" if not dict else {}

```

## General Usage