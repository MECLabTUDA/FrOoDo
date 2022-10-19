import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F

from .ood_methods import OODMethod


class MCDropout(OODMethod):
    def __init__(self, hyperparams={"dropout": 0.5, "samples": 50}) -> None:
        super().__init__(hyperparams)
        self.dropout = hyperparams["dropout"]
        self.samples = hyperparams["samples"]

    def get_params(self, dict=False):
        if dict:
            return {"dropout rate": self.dropout, "number of samples": self.samples}
        return f"dropout rate: {self.dropout}, number of samples: {self.samples}"

    def modify_net(self, net):
        feats_list = list(net.children())
        if True:  # Dropout at the end for inception_resnet_v2
            feats_list.insert(-1, nn.Dropout(p=self.dropout, inplace=True))
            net = nn.Sequential(*feats_list)
        else:  # Dropout after each Convolusion layer
            new_feats_list = []
            for feat in feats_list:
                new_feats_list.append(feat)
                if isinstance(feat, nn.Conv2d):
                    new_feats_list.append(nn.Dropout(p=self.dropout, inplace=True))
            net = nn.Sequential(*new_feats_list)

        return net

    def _enable_dropout(self, net):
        for m in net.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def _calculate_mean_of_samples(self, X, net, num_samples):
        if True:  # Special case for inception_resnet_v2
            net_before_dropout = nn.Sequential(*list(net.children())[:-2])
            net_after_dropout = nn.Sequential(*list(net.children())[-2:])
            with torch.no_grad():
                p = net_before_dropout(X.cuda())
            preds = []
            for _ in range(num_samples):
                with torch.no_grad():
                    res = net_after_dropout(p)
                preds.append(res)
        else:  # Otherwise
            preds = []
            for _ in range(num_samples):
                with torch.no_grad():
                    res = net(p)
                preds.append(res)
        return torch.stack(preds).mean(axis=0)

    def calculate_ood_score(self, imgs, net, batch=None):
        net = self.modify_net(net)
        self._enable_dropout(net)

        outputs = self._calculate_mean_of_samples(imgs, net, self.samples)
        outputs = self._post_proc_net_output(outputs)

        scores = torch.softmax(outputs[0].data, dim=1)
        m, _ = torch.max(scores, dim=1)
        return m.detach().cpu()
