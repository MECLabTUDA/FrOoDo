from ..artifacts.storage import ArtifactStorage


class Component:
    def __init__(self, overwrite_from_artifacts=True) -> None:
        self.artifacts = {}
        self.overwrite_from_artifacts = overwrite_from_artifacts

    def sanity_check(self) -> bool:
        raise NotImplementedError("Please Implement this method")

    def load_from_artifact_storage(self, storage: ArtifactStorage):
        raise NotImplementedError("Please Implement this method")

    def __call__(self):
        raise NotImplementedError("Please Implement this method")

    def get_artifacts(self):
        return self.artifacts
