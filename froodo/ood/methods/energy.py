import torch

from .ood_methods import OODMethod


class EnergyBased(OODMethod):
    def __init__(self, hyperparams={"temperature": 1}) -> None:
        super().__init__(hyperparams)
        self.temperature = hyperparams["temperature"]

    def get_params(self, dict=False):
        if dict:
            return {"temperature": self.temperature}
        return f"temp: {self.temperature}"

    def calculate_ood_score(self, imgs, net, batch=None):
        with torch.no_grad():
            output, *_ = net(imgs.cuda())
        scores = -self.temperature * torch.logsumexp(output / self.temperature, dim=1)
        return -scores.detach().cpu()
