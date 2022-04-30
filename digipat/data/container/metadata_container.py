import pandas as pd

from typing import List

from .container import Container
from .requirements import ContainerRequirements


class MetadataContainer(Container):
    def __init__(self) -> None:
        super().__init__()

    def data_requirements(self) -> List[ContainerRequirements]:
        return [ContainerRequirements.METADATA]

    def process(self):
        self.metaframe = pd.json_normalize([m.data for m in self.metadata])
