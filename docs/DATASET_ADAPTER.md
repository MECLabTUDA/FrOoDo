# Dataset adaptation

To make FrOoDo flexible and able to handle your specific problem with your dataset, it is possible to use any dataset as long as you use a dataset adapter. 

## SampleDataset
To use own datasets you need to adapt your dataset to a SampleDataset. This can be done by a dataset adapter which is essentially an own dataset with only the purpose to translate the dataset output from your dataset to the SampleDataset.

A SampleDataset is a FrOoDo specific dataset which returns an object of the class Sample. A Sample consist out of three objects: 

- The image: Image tensor. All augmentations will be applied on the image
- The label_dict: Default dictionary that contains information abut the labels e.g. the segmentation mask, the ood mask or the class label
- The metadata: The metadata object is like a default dict and saves relevant information about an image e.g. OOD status, ignore index, OOD reason



## Built-In Adapters

The easiest way to adapt your dataset is to use the GeneralDatasetAdapter and provide a remapping if necessary.

```python
# create remapping with helper function
remapping = create_mapping_dict(image=0, segmentation_mask=1)

# use GeneralDatasetAdapter and remapping to adapt any dataset
adapter = GeneralDatasetAdapter(my_dataset, remapping=remapping)

# test if adaptation worked
dataset.sample()
```

The GeneralAdapter will try to find the best adapter according to your inputs and your remapping. Of course you can also use one of the specialized Adapters:

```python
# use this if your dataset already returns a tuple of image, a label dictionary and optionally metadata
adapter = ImageLabelMetaAdapter(my_dataset)

# use this adapter if your dataset returns a tuple of tensors
adapter = PositionalAdapter(my_dataset, remapping = {"image": 0, "segmentation_mask": 1})

# use this adapter if your dataset returns a dictionary
adapter = DatasetWithAllInOneDictAdapter(my_dataset, remapping= {
    "image": "img",
    "segmentation_mask": "mask"
})
```

## Remapping

FrOoDo already has some standard remapping keys which should not be changed.

Keywords |  Type | Function | Usage in Component
 -- | -- | -- | --
image | torch.Tensor | image tensor | [OOD Evaluation](../froodo/experiment/components/interfaces/ood_strategy_component.py), Segmentation Performance
segmantation_mask | torch.Tensor  |segmentation mask of the image | Segmentation Performance
ood_mask | torch.Tensor  |OOD mask for the image (contains only 0,1 and ignore index) |  [OOD Evaluation](../froodo/experiment/components/interfaces/ood_strategy_component.py)
class_label |  | |  [OOD Evaluation](../froodo/experiment/components/interfaces/ood_strategy_component.py), Classification Performance
metadata | froodo.SampleMetadata | metadata object to save meta information about the image | [OOD Evaluation](../froodo/experiment/components/interfaces/ood_strategy_component.py)


## Create your own adapter

If none of the predefined Adapters should suit your dataset you can easily create an own adapter. The Adapter Class looks like this:


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