import torch

from .ood_methods import OODMethod


class MaxClassBaseline(OODMethod):
    def __init__(self, hyperparams={}) -> None:
        super().__init__(hyperparams)

    def calculate_ood_score(self, imgs, net, batch=None):
        with torch.no_grad():
            outputs, *_ = net(imgs.cuda())
        scores = torch.softmax(outputs.data, dim=1)
        m, _ = torch.max(scores, dim=1)
        return m.detach().cpu()
