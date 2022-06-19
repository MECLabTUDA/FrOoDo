from .artifact import ComponentArtifact
from ...data.container import Container


class ContainerArtifact(ComponentArtifact):
    def __init__(self, container: Container) -> None:
        super().__init__()
        self.container = container
