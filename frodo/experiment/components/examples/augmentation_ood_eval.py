from typing import List

from .. import OODStreatgyComponent
from ....models import Model
from ....ood.metrics import Metric
from ....ood.methods import OODMethod
from ....ood.strategies import AugmentationStrategy
from ....data.datasets.adapter import DatasetAdapter
from ....ood.augmentations import Augmantation


class AugmentationOODEvaluationComponent(OODStreatgyComponent):
    def __init__(
        self,
        data_adapter: DatasetAdapter,
        augmentation: Augmantation,
        model: Model,
        methods: List[OODMethod] = None,
        metrics: List[Metric] = None,
        seed: int = None,
    ) -> None:
        super().__init__(
            AugmentationStrategy(data_adapter, augmentation, seed=seed),
            model,
            methods,
            metrics,
            seed,
        )
