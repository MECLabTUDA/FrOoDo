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
            outputs = net(imgs.cuda())
        outputs = self._post_proc_net_output(outputs)
        scores = -self.temperature * torch.logsumexp(
            outputs[0] / self.temperature, dim=1
        )
        return -scores.detach().cpu()
