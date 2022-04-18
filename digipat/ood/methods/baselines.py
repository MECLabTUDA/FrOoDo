import torch

from .ood_methods import OODMethod


class MaxClassBaseline(OODMethod):
    def __init__(self, hyperparams={}) -> None:
        super().__init__(hyperparams)
        self.name = "MaxClass"

    def __call__(self, imgs, masks, net):
        with torch.no_grad():
            outputs, *_ = net(imgs.cuda())
        scores = torch.softmax(outputs.data, dim=1)
        m, _ = torch.max(scores, dim=1)
        return m.detach().cpu().numpy()
