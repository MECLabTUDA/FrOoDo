from typing import List


class Container:
    def data_requirements(self) -> List[str]:
        raise NotImplementedError("Please Implement this method")

    def append(self, data: dict):
        raise NotImplementedError("Please Implement this method")

    def process(self):
        raise NotImplementedError("Please Implement this method")
