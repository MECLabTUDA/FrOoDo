import torch
from torch.autograd import Variable

from .ood_methods import OODMethod


class ODIN(OODMethod):
    def __init__(self, hyperparams={"temperature": 2, "noise": 0.05}) -> None:
        super().__init__()
        self.temperature = hyperparams["temperature"]
        self.noise = hyperparams["noise"]
        self.criterion = torch.nn.CrossEntropyLoss()
        self.name = "ODIN"

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

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        tempInputs = torch.add(inputs.data, gradient, alpha=-self.noise)
        outputs, *_ = net(tempInputs)
        outputs = outputs / self.temperature
        return outputs

    def __call__(self, imgs, masks, net):
        outputs = self._temp_and_pertubate(imgs, net)
        outputs = outputs - torch.max(outputs, axis=1)[0].unsqueeze(1)
        scores = torch.softmax(outputs.data, dim=1)
        m, _ = torch.max(scores, axis=1)
        return m.detach().cpu().numpy()
