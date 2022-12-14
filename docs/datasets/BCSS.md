# How to setup BCSS dataset for FrOoDo's build-in BCSS interface

## Disclaimer
 It is **not mandatory to use the build-in dataset class** to use it within the framework. Therefore you can write your own dataset and use a dataset adapter. This docu is meant to quickly setup BCSS dataset to run the demo on it.

## Lets start

Please download the BCSS dataset [1] files following the the instructions of the github repository: https://github.com/PathologyDataScience/BCSS.

To use the build-in BCSS Dataset classes the files need to be converted into tif files. You can use the static method of the MultiFileOverlappingTilesDataset to do that.

```python
from froodo import *

MultiFileOverlappingTilesDataset.save_as_tif(
    raw_folder = "folder to downloaded raw files",
    tif_folder = "folder where you want the tif images"
)
```

Now that you have the files in the correct format, you can use the  MultiFileOverlappingTileDataset Class to load the BCSS Dataset. 

```python
bcss_dataset = MultiFileOverlappingTilesDataset(
        folder="path to tif folder",
        tile_folder="path where tile config is saved",
        size=(620, 620),
        dataset_name="bcss",
        mode=["mask", "full_ood"],
        overlap=0.2,
        map_classes=[],
        ignore_classes=[0, 7, 17],
        ood_classes=[5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21],
        remain_classes=[1, 2, 3, 4],
        ignore_index=5,
        non_ignore_threshold=0.9
)
```

The class also has a function to split the Dataset into train/val/test dataset by splitting the whole WSIs to ensure that no data appears in two splits at the same time.

```python
# split dataset along whole slide images
train_set, val_set, test_set = bcss_dataset.get_train_val_test_set()

# use an dataset adapter to use the dataset in the framework
bcss_adapted_test_set = GeneralDatasetAdapter(test_set)

# view samples
bcss_adapted_test_set.sample()
```

If you want to crop or resize the images you can use an Augmentation Pipeline:

```python
# build augmentation pipeline
pipeline = SizeInOODPipeline(in_augmentations=[
                InCrop((600, 600)),
                InResize(resize_size=(300, 300))
            ])

# create dataset that applies augmentations on samples
cropped_resized_bcss_set = AugmentationDataset(bcss_adapted_test_set, pipeline)

cropped_resized_bcss_set.sample()
```

## FAQ
Question | Answer
:-- | --
Why does it **take so long** to load the dataset the **first time**? | When you first load the Dataset class it will produce a file that contains all valid regions satisfying your parameters. With this file the dataset will lazy load all tile regions which is much faster than loading in the whole WSI and more flexible that saving all tiles individually


## References

[1] Amgad, M., Elfandy, H., Hussein, H., Atteya, L.A., Elsebaie, M.A., Abo Elnasr, L.S., Sakr, R.A., Salem, H.S., Ismail, A.F., Saad, A.M., et al.: Structured crowd- sourcing enables convolutional segmentation of histology images. Bioinformatics (2019)