import torch
from tqdm import tqdm
import numpy as np

from typing import List

from .. import Component
from ....data.metadata import SampleMetadataCommonTypes
from ....models import Model
from ....ood.metrics import infer_container, Metric
from ....ood.methods import OODMethod
from ....ood.strategies import OODStrategy
from ....data.container import ContainerRequirements


class OODStreatgyComponent(Component):
    def __init__(
        self,
        strategy: OODStrategy,
        methods: List[OODMethod],
        metrics: List[Metric],
        model: Model,
        seed: int = None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.methods = methods
        self.metrics = metrics
        self.model = model
        self.seed = seed

    def __call__(self):

        self.strategy.dataset.sample(10)

        # create data container based on the metric requirements
        container = infer_container(self.metrics)

        if self.seed != None:
            torch.manual_seed(1234)

        for batch in tqdm(
            self.strategy.get_dataloader(batch_size=16, num_workers=8, shuffle=True),
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
                c.append({ContainerRequirements.METADATA: batch["metadata"].batch})

        for c in container:
            c.process()

        for m in self.metrics:
            m(container, {})
            m.present()
