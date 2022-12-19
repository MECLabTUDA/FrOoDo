from examples.XrayDatasetAdapter import XrayDatasetAdapter
from froodo import *

############################################################

xray_dataset = XrayDatasetAdapter('~/Downloads/chest_xray/test')
# test if adaptation worked
xray_dataset.sample()

#############################################################

# init network
net = ResNet50(3, 9)
#net.load_state_dict(
    # load in your weights
#)
#net = net.cuda()

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
    augmentation=SampledAugmentation(MotionAugmentation(keep_ignored=False)),
    model=net,
    metrics=metrics,
    methods=methods,
    seed=4321,
    task_type=TaskType.CLASSIFICATION,
    batch_size=128,
    num_workers=0
)

# run experiment
experiment()
