from tokenize import group
from sklearn.metrics import auc, roc_auc_score
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

from typing import List

from .metrics import Metric, MetricGroup
from ...data.container import MetadataContainer, Container
from ...data.metadata import MetadataCommonTypes
from ...data.datatypes import DistributionSampleType


class OODAuRoC(MetricGroup):
    def __init__(self, group_by=None, **kwargs) -> None:
        super().__init__()
        self.metric = AUROCMetric()
        self.group_by = group_by
        if group_by == "bin":
            self.num_bins = kwargs.get("num_bins", 20)

    def requires(self) -> List[Container]:
        return [MetadataContainer]

    def __call__(self, data_container: List[Container], experiment_metadata: dict):
        metaframe: pd.DataFrame = data_container[0].metaframe.copy()
        self.score_colums = [
            col
            for col in metaframe
            if col.startswith(f"{MetadataCommonTypes.OOD_SCORE.name}.")
        ]
        ood_frame = metaframe[(metaframe["type"] == DistributionSampleType.OOD_DATA)]
        in_array = metaframe[(metaframe["type"] == DistributionSampleType.IN_DATA)][
            self.score_colums
        ].to_numpy()
        in_labels = np.ones((in_array.shape[0]))

        if self.group_by == None:

            ood_array = ood_frame[self.score_colums].to_numpy()
            ood_labels = np.zeros((len(ood_frame)))
            self.value = np.zeros((len(self.score_colums)))
            for i, _ in enumerate(self.score_colums):
                self.value[i] = self.metric(
                    np.concatenate([in_labels, ood_labels]),
                    np.concatenate([in_array[:, i], ood_array[:, i]]),
                )

        else:
            metaframe["bin"] = metaframe["OOD_SEVERITY"].apply(
                lambda x: x.get_bin(self.num_bins) if not pd.isna(x) else None
            )
            ood_frame = ood_frame.groupby("bin")
            ood_nested_list = (
                ood_frame[self.score_colums].apply(lambda x: x.values.tolist()).tolist()
            )
            bin_keys = ood_frame[self.score_colums].groups.keys()

    def present(self, **kwargs):
        if self.group_by == None:
            if kwargs.get("as_plot", False):
                plt.barh([s[12:] for s in self.score_colums], self.value)
                plt.xlim([min(self.value) - 0.05, max(self.value) + 0.05])
                plt.title("AUC Score for different OOD Methods")
                plt.show()
            else:
                print(
                    tabulate(
                        [[b, self.value[i]] for i, b in enumerate(self.score_colums)],
                        headers=["Name", "AUC"],
                    )
                )


class AUROCMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, labels, scores):
        self.value = roc_auc_score(
            labels[~np.isnan(labels)],
            scores[~np.isnan(scores)],
        )
        return self.value

    def present(self, **kwargs):
        print(f"AuRoC Score: {self.value}")
