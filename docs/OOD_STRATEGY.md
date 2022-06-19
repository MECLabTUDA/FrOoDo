# FrOoDo's Out-of-Distribution Strategies

An OOD Strategy is the way how the OOD Data is generated or determined. It is therefore a important building block of OOD experiments that rely on both In and OOD data.

Strategy |  OOD Data origin | Description
:-- | :-- | :--
Augmentation Stretegy |  OODAugmentation, that corrupts an In-Distribution image into an OOD image e.g. tissue artifacts | The augmentation stretgy creates an AugmentationDataset that determines by chance which image is augmented and which is left unchanged. 
OOD Dataset Strategy | A whole Dataset is being used as OOD data | In the OOD Dataset strategy the type of data (OOD or IN) is only defined by the dataset from which an image is being taken. All data from the OOD Dataset will be marked as OOD Data. Warning: The Data of the In distribution dataset will not be checked whether the images are truly In-Distribution.

## Build an own strategy

To build an own strategy you just need to implement a class that inherits the  OODStrategy class:

```python
class OODStrategy:
    def get_dataloader(self, **dataloader_kwargs):
        return DataLoader(self.dataset, collate_fn=sample_collate, **dataloader_kwargs)
```


