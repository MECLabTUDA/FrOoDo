import torch

from typing import Optional, Dict, Any

from .ood_methods import OODMethod
from ...data.samples import Batch


class GODIN_Max_of_h(OODMethod):
    def __init__(
        self,
        hyperparams: Dict[str, Any] = ...,
        rejection_threshold: Optional[float] = None,
    ) -> None:
        super().__init__(hyperparams, rejection_threshold)

    def calculate_ood_score(
        self, imgs: torch.Tensor, net: torch.nn.Module, batch: Optional[Batch] = None
    ):
        with torch.no_grad():
            outputs = net(imgs.cuda())

        outputs = self._post_proc_net_output(outputs)

        m, _ = torch.max(outputs[1].data, dim=1)
        return m.detach().cpu()


class GODIN_d(OODMethod):
    def __init__(
        self,
        hyperparams: Dict[str, Any] = ...,
        rejection_threshold: Optional[float] = None,
    ) -> None:
        super().__init__(hyperparams, rejection_threshold)

    def calculate_ood_score(
        self, imgs: torch.Tensor, net: torch.nn.Module, batch: Optional[Batch] = None
    ):
        with torch.no_grad():
            outputs = net(imgs.cuda())

        outputs = self._post_proc_net_output(outputs)

        m, _ = torch.max(outputs[2].data, dim=1)
        return m.detach().cpu()
