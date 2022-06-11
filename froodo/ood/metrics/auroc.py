from tokenize import group
from sklearn.metrics import auc, roc_auc_score
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

from typing import List
import inspect

from .metrics import Metric, MetricGroup
from ...data.container import MetadataContainer, Container
from ...data.metadata import SampleMetadataCommonTypes
from ...data.datatypes import DistributionSampleType
from ..severity import create_bins


class OODAuRoC(MetricGroup):
    def __init__(self, group_by=None, bin_by=None, **kwargs) -> None:
        super().__init__()
        self.metric = AUROCMetric()
        self.group_by = group_by
        self.bin_by = bin_by
        if self.bin_by != None:
            self.group_by = "bin"
        if self.group_by == "bin":
            self.num_bins = kwargs.get("num_bins", 20)

    def requires(self) -> List[Container]:
        return [MetadataContainer]

    def _group_by(self, ood_frame, in_labels, in_array):

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
            # print(metaframe)
            if self.bin_by != None:
                ood_frame["bin"] = ood_frame[self.bin_by].apply(
                    lambda x: x.get_bin(self.num_bins) if not pd.isna(x) else None
                )
                self.bin_range = ood_frame.iloc[0][self.bin_by].range
            ood_frame = ood_frame.groupby(self.group_by, sort=self.group_by == "bin")
            ood_nested_list = (
                ood_frame[self.score_colums].apply(lambda x: x.values.tolist()).tolist()
            )
            self.group_keys = ood_frame[self.score_colums].groups.keys()

            self.value = np.empty(
                (
                    len(self.score_colums),
                    len(self.group_keys) if self.group_by != "bin" else self.num_bins,
                )
            )
            self.value[:] = np.NaN

            for i, g_k in enumerate(self.group_keys):
                group_array = np.array(ood_nested_list[i])
                for j, _ in enumerate(self.score_colums):
                    ood_labels = np.zeros((group_array.shape[0]))
                    self.value[j, i if self.group_by != "bin" else g_k] = self.metric(
                        np.concatenate([in_labels, ood_labels]),
                        np.concatenate([in_array[:, j], group_array[:, j]]),
                    )

    def __call__(self, data_container: List[Container], experiment_metadata: dict):
        self.component_metadata = experiment_metadata
        metaframe: pd.DataFrame = data_container[0].metaframe.copy()
        self.score_colums = [
            col
            for col in metaframe
            if col.startswith(f"{SampleMetadataCommonTypes.OOD_SCORE.name}.")
        ]
        ood_frame = metaframe[
            (metaframe["type"] == DistributionSampleType.OOD_DATA)
        ].reset_index()
        in_array = metaframe[(metaframe["type"] == DistributionSampleType.IN_DATA)][
            self.score_colums
        ].to_numpy()
        in_labels = np.ones((in_array.shape[0]))

        self._group_by(ood_frame, in_labels, in_array)

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
                        [
                            [f"{b.display_name()} {b.get_params()}", self.value[i]]
                            for i, b in enumerate(self.component_metadata["methods"])
                        ],
                        headers=["Name", "AUC"],
                    )
                )
        else:
            if self.group_by == "bin":
                for i in range(len(self.score_colums)):
                    plt.plot(
                        create_bins(self.bin_range, self.num_bins),
                        self.value[i],
                        label=type(self.component_metadata["methods"][i]).__name__,
                    )
                    plt.xlabel(self.bin_by)
            else:
                for i in range(len(self.score_colums)):
                    plt.scatter(
                        np.array(list(range(len(self.group_keys)))),
                        self.value[i],
                        label=type(self.component_metadata["methods"][i]).__name__,
                        s=100,
                        alpha=0.8,
                    )
                plt.xlabel(self.group_by)
                plt.xlim([-0.5, len(self.group_keys) - 0.5])
                plt.xticks(
                    list(range(len(self.group_keys))),
                    [s.__name__ if inspect.isclass(s) else s for s in self.group_keys],
                    rotation=45,
                )

            plt.ylabel("AuRoC")
            plt.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                fancybox=True,
                shadow=True,
                ncol=5,
            )
            plt.show()


class AUROCMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, labels, scores):
        self.value = roc_auc_score(
            labels[~np.isnan(scores)],
            scores[~np.isnan(scores)],
        )
        return self.value

    def present(self, **kwargs):
        print(f"AuRoC Score: {self.value}")
