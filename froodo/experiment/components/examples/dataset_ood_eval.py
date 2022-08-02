from typing import List

from .. import OODStreatgyComponent
from ....models import Model
from ....ood.metrics import Metric
from ....ood.methods import OODMethod
from ....ood.strategies import OODDatasetsStrategy
from ....data.datasets.adapter import DatasetAdapter
from ....ood.augmentations import Augmentation
from ....data.datatypes import TaskType


class DatasetOODEvaluationComponent(OODStreatgyComponent):
    def __init__(
        self,
        in_dataset_list: List[DatasetAdapter],
        ood_dataset_list: List[DatasetAdapter],
        model: Model,
        methods: List[OODMethod] = None,
        metrics: List[Metric] = None,
        task_type=TaskType.SEGMENTATION,
        seed: int = None,
        remove_ignore_index: bool = False,
        show: bool = True,
        batch_size: int = 12,
        num_workers: int = 8,
        shuffle: bool = True,
    ) -> None:
        super().__init__(
            OODDatasetsStrategy(in_dataset_list, ood_dataset_list),
            model,
            methods,
            metrics,
            seed,
            task_type=task_type,
            remove_ignore_index=remove_ignore_index,
            show=show,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
