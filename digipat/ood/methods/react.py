import torch
from torch.utils.data import DataLoader
import tqdm
import numpy as np

import pickle

from .ood_methods import OODMethod


class ReAct(OODMethod):
    def __init__(self, hyperparams={"quantile": 0.95}) -> None:
        super().__init__(hyperparams)
        self.name = "ReAct"
        with open("react.pkl", "rb") as f:
            quantiles = pickle.load(f)
        self.q = round(hyperparams["quantile"], 2)
        self.tensor = quantiles[self.q].cuda()

    def generate_quantiles(
        self, dataset, net, range_vals=np.arange(0.7, 1, 0.01), name="react"
    ):
        loader = DataLoader(dataset, batch_size=16)
        data_iter = tqdm.tqdm(loader)
        data_iter.set_description(f"Loading")
        activations = []
        for i, (images, masks) in enumerate(data_iter):
            _, *act = net(images.cuda())
            activations.append(
                act[-1].permute(0, 2, 3, 1).reshape(-1, 16).detach().cpu()[::20]
            )

        out = torch.cat(activations, axis=0)

        quantiles = {}

        for q in range_vals:
            q = round(q, 2)
            quantiles[q] = torch.quantile(out, q, 0)

        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(quantiles, f)

        return quantiles
