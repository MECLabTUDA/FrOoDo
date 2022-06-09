# FrODo - Framework for Out of Distribution Detection

## Introduction


## Installation

```
git clone ...
cd frodo
pip install -r requirements.txt
```

## Demo

```python
from frodo.quickstart import *

# init network
net = SegmentationModel().load()

# choose metrics
metrics = [
    OODAuRoC(bin_by='OOD_SEVERITY', num_bins=50),
    OODAuRoC(),
]

# choose post-hoc OOD metods
methods = [MaxClassBaseline(), ODIN(), EnergyBased()]

# create Experiment Compoonent
component = AugmentationOODEvaluationComponent(
    data_adapter=BCSS_Adapted_Cropped_Resized_Datasets().val,
    augmentation=SampledAugmentation(DarkSpotsAugmentation()),
    model=net,
    metrics=metrics,
    methods=methods
    seed=4321,
)

# run experiment
component()
```


## Manual

<style>
    table {
        width: 100%;
    }
</style>

Explanation | Link
-- | :--:
Create new <b>Post-hoc OOD methods</b> | [here](docs/NEW_METHOD.md)
Create new <b>metrics</b> for your experiments | [here](docs/NEW_METRIC.md)
Create new <b>augmentation</b> for your dataset| tbd
Dataset Adaptation | tbd
How does the framework works internally?| tbd




