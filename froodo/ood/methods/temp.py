import torch
from torch.autograd import Variable
from torch import optim, nn

from .ood_methods import OODMethod


class TemperatureScaling(OODMethod):
    def __init__(self, hyperparams={"temperature": 2.0}) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(hyperparams["temperature"]))
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def display_name(self) -> str:
        return f"{super().display_name()} {self.get_params()}"

    def get_params(self, dict=False):
        if dict:
            return {"temperature": self.temperature.item()}
        return f"temp: {self.temperature.item()}"

    def _temperature_scale(self, logits):
        return logits / self.temperature.cuda()

    def _temp(self, imgs, net):
        optimizer = optim.Adam([self.temperature], lr=1e-3)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.1
        )

        optimizer.zero_grad()
        outputs = net(imgs.cuda())
        outputs = self._post_proc_net_output(outputs)

        max = torch.max(outputs[0].data, axis=1)[0]
        scores = torch.softmax(outputs[0].data - max.unsqueeze(1), dim=1)
        labels = torch.argmax(scores, axis=1)

        loss = self.criterion(self._temperature_scale(outputs[0]), labels)
        loss.backward()
        if loss.item()>1e-5:
            optimizer.step()
            lr_scheduler.step(loss)

        outputs = outputs[0] / self.temperature.cuda()
        return outputs

    def calculate_ood_score(self, imgs, net, batch=None):
        outputs = self._temp(imgs, net)
        outputs = outputs - torch.max(outputs, axis=1)[0].unsqueeze(1)
        scores = torch.softmax(outputs.data, dim=1)
        m, _ = torch.max(scores, axis=1)
        return m.detach().cpu()
