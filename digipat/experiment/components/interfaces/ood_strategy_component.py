import torch
from tqdm import tqdm
import numpy as np

from typing import List

from ..metadata import ComponentMetadata
from .. import Component
from ...artifacts.metric_artifact import MetricArtifact
from ....data.datasets.interfaces.utility_dataset import SampleDataset
from ....data.metadata import SampleMetadataCommonTypes
from ....models import Model
from ....ood.metrics import infer_container, Metric, OODAuRoC
from ....ood.methods import OODMethod, MaxClassBaseline, ODIN, EnergyBased
from ....ood.strategies import OODStrategy


class OODStreatgyComponent(Component):
    def __init__(
        self,
        strategy: OODStrategy,
        model: Model,
        methods: List[OODMethod] = None,
        metrics: List[Metric] = None,
        seed: int = None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.model = model
        self.methods = methods
        if self.methods == None:
            self.methods = [MaxClassBaseline(), ODIN(), EnergyBased()]
        self.metrics = metrics
        if self.metrics == None:
            self.metrics = [OODAuRoC()]
        self.seed = seed
        self.metadata = ComponentMetadata({})

    def sanity_check(self):
        return super().sanity_check()

    def __call__(self):

        if isinstance(self.strategy.dataset, SampleDataset):
            self.strategy.dataset.sample(10)

        # create data container based on the metric requirements
        container = infer_container(self.metrics)

        if self.seed != None:
            torch.manual_seed(1234)

        for batch in tqdm(
            self.strategy.get_dataloader(batch_size=16, num_workers=8, shuffle=False),
            "OOD Evaluation",
        ):

            for index, m in enumerate(self.methods):
                scores = m(batch["image"], batch["ood_mask"], self.model)
                scores = np.mean(scores, tuple(range(1, scores.ndim)))
                batch["metadata"].append_to_keyed_dict(
                    SampleMetadataCommonTypes.OOD_SCORE.name,
                    f"{index}_{type(m).__name__}",
                    scores,
                )

            for c in container:
                c.append({"BATCH": batch})

        for c in container:
            c.process()

        for m in self.metrics:
            m(container, {})
            m.present()

    def create_artifact(self):
        return MetricArtifact(self)
