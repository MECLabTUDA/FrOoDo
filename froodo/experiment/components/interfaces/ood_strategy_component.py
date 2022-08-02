import torch
from tqdm import tqdm
import numpy as np

from typing import List


from ..metadata import ComponentMetadata
from .. import Component
from ...artifacts import MetricArtifact, ContainerArtifact
from ....data.datasets.interfaces.sample_dataset import SampleDataset
from ....data.metadata import SampleMetadataCommonTypes
from ....data.datatypes import TaskType
from ....models import Model
from ....ood.metrics import infer_container, Metric, OODAuRoC
from ....ood.methods import OODMethod, MaxClassBaseline, ODIN, EnergyBased
from ....ood.strategies import OODStrategy
from ....experiment.artifacts.storage import ArtifactStorage


class OODStreatgyComponent(Component):
    def __init__(
        self,
        strategy: OODStrategy,
        model: Model,
        methods: List[OODMethod] = None,
        metrics: List[Metric] = None,
        seed: int = None,
        task_type: TaskType = TaskType.SEGMENTATION,
        remove_ignore_index: bool = False,
        show: bool = False,
        overwrite_from_artifacts: bool = True,
        batch_size: int = 12,
        num_workers: int = 8,
        shuffle: bool = True,
    ) -> None:
        super().__init__(overwrite_from_artifacts)
        self.strategy = strategy
        self.model = model
        self.methods = methods
        if self.methods == None:
            self.methods = [MaxClassBaseline(), ODIN(), EnergyBased()]
        self.metrics = metrics
        if self.metrics == None:
            self.metrics = [OODAuRoC()]
        self.seed = seed
        assert isinstance(task_type, TaskType)
        self.task_type = task_type
        self.show = show
        self.remove_ignore_index = remove_ignore_index
        self.metadata = ComponentMetadata(
            {"strategy": self.strategy, "methods": self.methods, "seed": self.seed}
        )
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle

    def sanity_check(self):
        return super().sanity_check()

    def load_from_artifact_storage(self, storage: ArtifactStorage):
        self.methods.extend(storage.get_latest("methods"))

    def __call__(self):

        self.model.eval()

        if isinstance(self.strategy.dataset, SampleDataset) and self.show:
            self.strategy.dataset.sample(7)

        # create data container based on the metric requirements
        container = infer_container(self.metrics)

        if self.seed != None:
            torch.manual_seed(self.seed)

        for batch in tqdm(
            self.strategy.get_dataloader(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
            ),
            "OOD Evaluation",
        ):

            for m in self.methods:
                batch = m(
                    batch.cuda(),
                    self.model,
                    task_type=self.task_type,
                    remove_ignore_index=self.remove_ignore_index,
                )

            for c in container:
                c.append({"BATCH": batch})

        for c in container:
            c.process()

        for m in self.metrics:
            m(container, self.metadata)
            if self.show:
                m.present()

        self.artifacts["container"] = [ContainerArtifact(c) for c in container]
        self.artifacts["metrics"] = [MetricArtifact(m) for m in self.metrics]
