from typing import List

from .. import OODStreatgyComponent
from ....models import Model
from ....ood.metrics import Metric
from ....ood.methods import OODMethod
from ....ood.strategies import OODDatasetsStrategy
from ....data.datasets.adapter import DatasetAdapter
from ....ood.augmentations import Augmantation


class DatasetOODEvaluationComponent(OODStreatgyComponent):
    def __init__(
        self,
        in_dataset_list: List[DatasetAdapter],
        ood_dataset_list: List[DatasetAdapter],
        model: Model,
        methods: List[OODMethod] = None,
        metrics: List[Metric] = None,
        seed: int = None,
    ) -> None:
        super().__init__(
            OODDatasetsStrategy(in_dataset_list, ood_dataset_list),
            model,
            methods,
            metrics,
            seed,
        )
