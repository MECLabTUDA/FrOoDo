from . import AllComposite, NTimesAugmentation, PickNComposite


class Buildable:
    def __init__(self) -> None:
        pass

    def build(self):
        raise NotImplementedError("Please Implement this method")


class AugmentationBuilder(Buildable):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = []

    def __add__(self, block):
        self.blocks.append(block.build() if isinstance(block, Buildable) else block)
        return self

    def build(self):
        return AllComposite(self.blocks)


class N(Buildable):
    def __init__(self) -> None:
        super().__init__()
        self.augmentations = []
        self.n = 1

    def __add__(self, aug):
        self.augmentations.append(aug)
        return self

    def __mul__(self, times):
        self.n *= times
        return self

    def build(self):
        return NTimesAugmentation(AllComposite(self.augmentations), self.n)


class Probability(Buildable):
    def __init__(self, replace=False) -> None:
        super().__init__()
        self.augmentations = []
        self.n = 1
        self.replace = replace
        self.probabilities = None

    def __add__(self, aug):
        self.augmentations.append(aug)
        return self

    def __mul__(self, times):
        self.n *= times
        return self

    def __truediv__(self, probabilities):
        self.probabilities = probabilities
        return self

    def build(self):
        return PickNComposite(
            self.augmentations,
            n=self.n,
            replace=self.replace,
            probabilities=self.probabilities,
        )
