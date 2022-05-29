import pandas as pd

from typing import List

from .container import Container


class MetadataContainer(Container):
    def __init__(self) -> None:
        super().__init__()
        self.metadata = []

    def data_requirements(self) -> List[str]:
        return ["BATCH"]

    def append(self, data: dict):
        for key, val in data.items():
            if key == "BATCH":
                self.metadata.extend(val.metadata.batch)

    def process(self):
        self.metaframe = pd.json_normalize([m.data for m in self.metadata])
