import torch

from .utils import *
from .examples import UNet

class SegmentationModel:
    def __init__(
        self,
        folder="UNet_noExtra_trainable_n16_d4_ADAM_bs16_lr0.0001_s20_g1_21",
        net_class=UNet,
        net_args={},
        state_dict_modification=rename_state_dict_keys,
    ) -> None:
        self.folder = folder
        self.state_dict_modification = state_dict_modification
        self.net_class = net_class
        self.net_args = net_args

    def load(self):
        net = self.net_class(**self.net_args)

        # rename stat dict
        wrong_dict = torch.load(f"U:\\ssl4uc\\saved\\{self.folder}\\model_best_val.pt")[
            "state_dict"
        ]
        correct_dict = self.state_dict_modification(wrong_dict)
        net.load_state_dict(correct_dict)

        net.cuda()
        net.eval()
        return net