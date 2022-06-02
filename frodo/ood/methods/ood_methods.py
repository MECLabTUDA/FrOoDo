import numpy as np
import torch

from typing import Callable

from ...data.samples import Sample, Batch
from ...data.datatypes import TaskType
from ...data.metadata.types import SampleMetadataCommonTypes


class OODMethod:
    def __init__(self, hyperparams={}) -> None:
        pass

    def get_params(self, dict=False):
        return "no params" if not dict else {}

    def display_name(self) -> str:
        return f"{type(self).__name__}"

    def modify_net(self, net):
        return net

    def remodify_net(self, net):
        return net

    def calculate_ood_score(self, imgs, net, batch=None):
        raise NotImplementedError("Please Implement this method")

    def _set_metadata(self, batch: Batch, scores) -> Batch:
        batch["metadata"].append_to_keyed_dict(
            SampleMetadataCommonTypes.OOD_SCORE.name,
            self.display_name(),
            scores,
        )
        return batch

    def __call__(
        self,
        batch: Batch,
        net,
        task_type: TaskType = TaskType.SEGMENTATION,
        remove_ignore_index: bool = True,
        score_reduction_method: Callable = np.mean,
    ) -> Sample:
        # modify network if needed
        net = self.modify_net(net)

        scores = self.calculate_ood_score(batch.image, net, batch).numpy()

        # if task type is segmentation ood score will be array with shape like the mask
        # therefore it will be reduced to one score per sample
        if task_type is TaskType.SEGMENTATION:

            # remove pixels which are assigned to the ignore class since it is not know wheather they are in oder ood
            if remove_ignore_index:
                non_ignore = batch["ood_mask"].numpy() == np.expand_dims(
                    np.array(batch.metadata["ignore_index"]),
                    list(range(1, batch["ood_mask"].ndim)),
                )

                scores = score_reduction_method(
                    np.ma.masked_array(scores, non_ignore),
                    axis=tuple(range(1, scores.ndim)),
                ).filled(np.nan)

            else:
                scores = score_reduction_method(
                    scores,
                    axis=tuple(range(1, scores.ndim)),
                )

        # save scores in metadata to later be accessed by the metrics
        batch = self._set_metadata(batch, scores)

        # always remodify network so it works properly for other mothods
        self.remodify_net(net)

        return batch
