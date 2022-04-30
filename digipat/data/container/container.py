from typing import List
from .requirements import ContainerRequirements


class Container:
    def __init__(self) -> None:
        if ContainerRequirements.METADATA in self.data_requirements():
            self.metadata = []

    def data_requirements(self) -> List[ContainerRequirements]:
        raise NotImplementedError("Please Implement this method")

    def append(self, data: dict):
        for key, val in data.items():
            if key == ContainerRequirements.METADATA:
                self.metadata.extend(val)

    def process(self):
        raise NotImplementedError("Please Implement this method")
