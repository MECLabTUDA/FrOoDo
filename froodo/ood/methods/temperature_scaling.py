import torch
from torch.autograd import Variable
from torch import optim, nn
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')

from typing import Optional, Dict, List

from .ood_methods import OODMethod
from ...data.datatypes import TaskType

class TemperatureScaling(OODMethod):
    def __init__(self, 
            hyperparams={"temperature": 2.0, "optimize": True},
            rejection_threshold: Optional[float] = None,
            requires_ood_samples_to_fit: bool = False,
            hyper_tuning_dict: Optional[Dict[str, List[float]]] = None,
            ignore_index: int = -100,
            ) -> None:
        super().__init__(
            hyperparams,
            rejection_threshold,
            requires_ood_samples_to_fit,
            hyper_tuning_dict,
        )
        self.temperature = nn.Parameter(torch.tensor(hyperparams.get("temperature", 0.007)).float())
        self.optimize_t = hyperparams.get("optimize", True)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def display_name(self) -> str:
        return f"{super().display_name()} {self.get_params()}"

    def get_params(self, dict=False):
        if dict:
            return {"temperature": self.temperature.item()}
        return f"temp: {self.temperature.item()}"

    def _temperature_scale(self, logits):
        return logits / self.temperature#.cuda()

    def _temp(self, imgs, net):
        with torch.no_grad():
            outputs = net(imgs.cuda())
            outputs = self._temperature_scale(outputs)
            #outputs = self._post_proc_net_output(outputs)
        return outputs

    def fit(self, model: torch.nn.Module, dataloader: DataLoader, task_type: TaskType, remove_ignore_index: bool = False) -> OODMethod:
        if self.optimize_t:
            logits_list = []
            labels_list = []
            index = 1804
            with torch.no_grad():
                for batch in dataloader:
                    (input, label) = (batch["image"], batch.get_label_by_task_type(task_type))
                    input = input.cuda()
                    logits = model(input)
                    logits_list.append(logits.detach().cpu())
                    labels_list.append(label.detach().cpu())
                if task_type is TaskType.SEGMENTATION:
                    logits = torch.cat(logits_list)[:index].detach().cpu()
                    labels = torch.cat(labels_list)[:index].detach().cpu()
                    self.temperature.cpu()
                    self.criterion.cpu()
                else:
                    logits = torch.cat(logits_list).cuda()#.float()#.squeeze_()
                    labels = torch.cat(labels_list).cuda()#.float()
                    self.temperature.cuda()
                    self.criterion.cuda()
            print(logits.size(),labels.size())
            # Calculate NLL before temperature scaling
            before_temperature_nll = self.criterion(logits, labels).item()
            print('Before temperature - NLL: %.3f' % (before_temperature_nll))

            # Next: optimize the temperature w.r.t. NLL
            optimizer = optim.LBFGS([self.temperature], lr=0.2*self.temperature.item(), max_iter=50)
            def eval():
                optimizer.zero_grad()
                loss = self.criterion(self._temperature_scale(logits), labels)
                loss.backward()
                return loss
            optimizer.step(eval)

            print(logits.size(),labels.size())
            # Calculate NLL after temperature scaling
            after_temperature_nll = self.criterion(self._temperature_scale(logits), labels).item()
            print('Optimal temperature: %.3f' % self.temperature.item())
            print('After temperature - NLL: %.3f' % (after_temperature_nll))
            if self.hyper_tuning_dict is not None:
                self.hyper_tuning_dict["temperature"] = [self.temperature]
        return super().fit(model,dataloader,task_type,bool)

    def calculate_ood_score(self, imgs, net, batch=None):
        outputs = self._temp(imgs, net)
        outputs = outputs - torch.max(outputs, axis=1)[0].unsqueeze(1)
        scores = torch.softmax(outputs.data, dim=1)
        m, _ = torch.max(scores, axis=1)
        return m.detach().cpu()
