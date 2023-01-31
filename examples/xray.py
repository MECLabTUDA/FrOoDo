from torchvision.models import ResNet50_Weights, resnet50

from examples.XrayDatasetAdapter import XrayDatasetAdapter
from froodo import *

############################################################

xray_dataset = XrayDatasetAdapter('~/Downloads/chest_xray/test')

#############################################################

# init network
net = ResNet18(3, 2).cuda()

weights = ResNet50_Weights.DEFAULT
net = resnet50(weights=weights)

# choose metrics
metrics = [
    OODAuRoC(bin_by='OOD_SEVERITY', num_bins=50),
    OODAuRoC(),
]

# choose post-hoc OOD methods
methods = [MaxClassBaseline(), ODIN(), EnergyBased()]

# create experiment component
experiment = AugmentationOODEvaluationComponent(
    data_adapter=xray_dataset,
    augmentation=SampledAugmentation(TubesAugmentation(keep_ignored=False)),
    model=net,
    metrics=metrics,
    methods=methods,
    seed=4321,
    task_type=TaskType.CLASSIFICATION,
    batch_size=64,
    num_workers=0
)

# run experiment
experiment()
