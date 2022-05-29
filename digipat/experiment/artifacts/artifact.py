from typing import List


class ComponentArtifact:
    def __init__(self) -> None:
        pass

    def save(self, folder) -> None:
        pass

    def recreate(self) -> None:
        pass

    @staticmethod
    def aggregate(artifacts: list, **kwargs):
        pass
