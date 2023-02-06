from froodo import *

# load dataset
xray_dataset = PneumoniaDataSetAdapter('~/Downloads/chest_xray/', split='test')

# init network
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False, num_classes=2)
model.load_state_dict(torch.load("./model.pth"))
model = model.cuda()

# choose metrics
metrics = [
    OODAuRoC(bin_by='OOD_SEVERITY', num_bins=100),
    OODAuRoC(),
]

# choose post-hoc OOD methods
methods = [MaxClassBaseline(), ODIN(), EnergyBased()]

# create experiment component
experiment = AugmentationOODEvaluationComponent(
    data_adapter=xray_dataset,
    augmentation=SampledAugmentation(TubesAugmentation(keep_ignored=False)),
    model=model,
    metrics=metrics,
    methods=methods,
    seed=4321,
    task_type=TaskType.CLASSIFICATION,
    batch_size=64,
    num_workers=0
)

# run experiment
experiment()
