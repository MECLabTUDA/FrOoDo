from typing import List
import cloudpickle
from os.path import join


class ComponentArtifact:
    def __init__(self) -> None:
        pass

    def save(self, folder) -> None:
        with open(join(folder, f"{type(self).__name__}.pkl"), "wb") as f:
            cloudpickle.dump(self, f)

    def recreate(self) -> None:
        pass
