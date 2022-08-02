from black import out
import torch

from .ood_methods import OODMethod


class MaxClassBaseline(OODMethod):
    def __init__(self, hyperparams={}) -> None:
        super().__init__(hyperparams)

    def calculate_ood_score(self, imgs, net, batch=None):
        with torch.no_grad():
            outputs = net(imgs.cuda())

        outputs = self._post_proc_net_output(outputs)

        scores = torch.softmax(outputs[0].data, dim=1)
        m, _ = torch.max(scores, dim=1)
        return m.detach().cpu()
