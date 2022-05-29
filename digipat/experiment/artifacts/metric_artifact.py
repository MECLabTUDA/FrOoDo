from .artifact import ComponentArtifact
from ...ood.metrics import Metric


class MetricArtifact(ComponentArtifact):
    def __init__(self, metric: Metric) -> None:
        super().__init__()
        self.metric = metric