# Augmentations

```python
class Augmentation:
    def _augment(self, sample: Sample) -> Sample:
        raise NotImplementedError("Please Implement this method")

    def _set_metadata(self, sample: Sample) -> Sample:
        return sample

    def __call__(self, sample: Sample) -> Sample:
        sample = init_augmentation(sample)
        sample = self._augment(sample)
        sample = self._set_metadata(sample)
        return sample
```

## Base types of augmentations

Interface | Description | Metadata Changes | Code
 -- | -- | -- | :--:
 Augmentation | Base Augmentation class from which every Augmentation inherits | None | [here](../froodo/ood/augmentations/types.py)
 OODAugmentation | Augmentation tht transforms ID images into OOD images e.g. artifacts | OOD Flag, Augmentation severity if provided | [here](../froodo/ood/augmentations/types.py)
 INAugmentation | Augmentation that does **not** change the OOD flag but keeps the data IN Distribution e.g. cropping or resizing of images | None | [here](../froodo/ood/augmentations/types.py)
 AugmentationComposite | An Augmentation composite is a combination of multiple augmentations e.g. pick N of a list | Dependent of parameters | [here](../froodo/ood/augmentations/types.py)

## Special types of augmentations

Interface | Description | Metadata Changes | Code
 -- | -- | -- | :--:
 SampableAugmentation | Sample Augmentation have sampable parameters which makes it possible to run an experiment with augmentations with different parameter values e.g. different scales for artifacts | Keeps Changes of base augmentation | [here](../froodo/ood/augmentations/types.py)
 ArtifactAugmentation | An Artfact Augmentation is a realization of an OODAugmentation and therefore inherits grom it. This interface comes with a method that overlays an artifact on the image | like OODAugmentation | [here](../froodo/ood/augmentations/pathology/artifacts/artifacts.py)

 ## FrOoDos Augmentations

 ![](../imgs/augmentations.png "Augmentation Samples with different scales/intensities") 

 Name |  Short description | Interface | Code 
 -- | -- | -- | --
 DarkSpotsAugmentation |  Creates a dark spot on the image |  ArtifactAugmentation, SampableAugmentation, OODAugmentation | [here](../froodo/ood/augmentations/pathology/artifacts/dark_spots.py)
FatAugmentation | Creates a fat drop on the image |  ArtifactAugmentation, SampableAugmentation, OODAugmentation | [here](../froodo/ood/augmentations/pathology/artifacts/fat.py)
SquamousAugmentation | Creates a tissue piece on the image |  ArtifactAugmentation, SampableAugmentation, OODAugmentation | [here](../froodo/ood/augmentations/pathology/artifacts/squamos.py)
ThreadAugmentation | Creates a thread on the image |  ArtifactAugmentation, SampableAugmentation, OODAugmentation | [here](../froodo/ood/augmentations/pathology/artifacts/threads.py)
GaussianBlurAugmentation | Blures the image |   SampableAugmentation, OODAugmentation | [here](../froodo/ood/augmentations/patchwise/gauss_blur.py)
ZoomInAugmentation | Crops the image and then resizes it back to original size |  SampableAugmentation, OODAugmentation | [here](../froodo/ood/augmentations/patchwise/crop.py)
BrightnessAugmentation | Changes brightness of image |  SampableAugmentation, OODAugmentation | [here](../froodo/ood/augmentations/patchwise/brightness.py)
InCrop | Crops image|   INAugmentation | [here](../froodo/ood/augmentations/indistribution/crop.py)
InResize | Resizes image |  INAugmentation | [here](../froodo/ood/augmentations/indistribution/resize.py)

## References
[1] Schömig-Markiefka, B., Pryalukhin, A., Hulla, W., Bychkov, A., Fukuoka, J., Mad- abhushi, A., Achter, V., Nieroda, L., Büttner, R., Quaas, A., et al.: Quality control stress test for deep learning-based diagnostic model in digital pathology. Modern Pathology (2021)