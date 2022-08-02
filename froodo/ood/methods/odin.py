import torch
from torch.autograd import Variable

from .ood_methods import OODMethod


class ODIN(OODMethod):
    def __init__(self, hyperparams={"temperature": 2, "noise": 0.004}) -> None:
        super().__init__()
        self.temperature = hyperparams.get("temperature", 2)
        self.noise = hyperparams.get("noise", 0.004)
        self.criterion = torch.nn.CrossEntropyLoss()

    def display_name(self) -> str:
        return f"{super().display_name()} {self.get_params()}"

    def get_params(self, dict=False):
        if dict:
            return {"temperature": self.temperature, "noise": self.noise}
        return f"temp: {self.temperature}, noise: {self.noise}"

    def _temp_and_pertubate(self, imgs, net):
        inputs = Variable(imgs.cuda(), requires_grad=True)
        outputs = net(inputs)
        outputs = self._post_proc_net_output(outputs)

        max = torch.max(outputs[0].data, axis=1)[0]
        scores = torch.softmax(outputs[0].data - max.unsqueeze(1), dim=1)

        labels = torch.argmax(scores, axis=1)
        loss = self.criterion(outputs[0] / self.temperature, labels)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        tempInputs = torch.add(inputs.data, gradient, alpha=-self.noise)
        outputs = net(tempInputs)
        outputs = self._post_proc_net_output(outputs)
        outputs = outputs[0] / self.temperature
        return outputs

    def calculate_ood_score(self, imgs, net, batch=None):
        outputs = self._temp_and_pertubate(imgs, net)
        outputs = outputs - torch.max(outputs, axis=1)[0].unsqueeze(1)
        scores = torch.softmax(outputs.data, dim=1)
        m, _ = torch.max(scores, axis=1)
        return m.detach().cpu()
