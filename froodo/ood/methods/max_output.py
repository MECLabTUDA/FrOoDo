import torch

from typing import Optional

from .ood_methods import OODMethod
from ...data.samples import Batch


class MaxOutput(OODMethod):
    def __init__(
        self,
        rejection_threshold: Optional[float] = None,
        return_max_logit: bool = False,
    ) -> None:
        super().__init__({}, rejection_threshold)
        self.return_max_logit = return_max_logit

    def display_name(self) -> str:
        return (
            f"{super().display_name()}_{'logits' if self.return_max_logit else 'soft'}"
        )

    def calculate_ood_score(
        self, imgs: torch.Tensor, net: torch.nn.Module, batch: Optional[Batch] = None
    ):
        with torch.no_grad():
            outputs = net(imgs.cuda())

        outputs = self._post_proc_net_output(outputs)

        if self.return_max_logit:
            m, _ = torch.max(outputs[0].data, dim=1)
        else:
            scores = torch.softmax(
                outputs[0],
                dim=1,
            )
            m, _ = torch.max(scores.data, dim=1)

        return m.detach().cpu()
