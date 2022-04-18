class OODMethod:
    def __init__(self, hyperparams={}) -> None:
        self.name = "set name"

    def __call__(self, imgs, masks, net):
        raise NotImplementedError("Please Implement this method")

    def modify_net(self, net):
        return net

    def get_params(self, dict=False):
        return "no params" if not dict else {}
