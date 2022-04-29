from sklearn.metrics import auc, roc_auc_score
import numpy as np

from .metrics import Metric


class AUROCMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, labels, scores):
        self.value = roc_auc_score(
            labels[~np.isnan(labels)],
            scores[~np.isnan(scores)],
        )
        return self.value

    def present(self):
        print(f"AuRoC Score: {self.value}")
