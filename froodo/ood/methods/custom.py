import torch
from torch.autograd import Variable

from .ood_methods import OODMethod
from .odin import ODIN


class TestOOD(OODMethod):
    def __init__(self, hyperparams={"temperature": 2, "noise": 0.05}) -> None:
        super().__init__()
        self.temperature = hyperparams["temperature"]
        self.noise = hyperparams["noise"]
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_params(self, dict=False):
        if dict:
            return {"temperature": self.temperature, "noise": self.noise}
        return f"temp: {self.temperature}, noise: {self.noise}"

    def _temp_and_pertubate(self, imgs, net):
        inputs = Variable(imgs.cuda(), requires_grad=True)
        outputs, *_ = net(inputs)

        max = torch.max(outputs.data, axis=1)[0]
        scores = torch.softmax(outputs.data - max.unsqueeze(1), dim=1)

        labels = torch.argmax(scores, axis=1)
        loss = self.criterion(outputs / self.temperature, labels)
        loss.backward()

        return torch.log(1 / (torch.abs(inputs.grad.data) + 1e-10))

    def calculate_ood_score(self, imgs, net, batch=None):
        outputs = self._temp_and_pertubate(imgs, net)
        return outputs.detach().cpu()
