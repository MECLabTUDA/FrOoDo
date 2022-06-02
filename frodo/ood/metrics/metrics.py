from typing import List

from ...data.container import Container


class MetricGroup:
    def __init__(self) -> None:
        pass

    def requires(self) -> List[Container]:
        raise NotImplementedError("Please Implement this method")

    def __call__(self, data_container: List[Container], experiment_metadata: dict):
        raise NotImplementedError("Please Implement this method")

    def present(self, **kwargs):
        pass


class Metric:
    def __init__(self) -> None:
        pass

    def __call__(self, data_container):
        raise NotImplementedError("Please Implement this method")

    def present(self, **kwargs):
        pass


def infer_container(metrics: List[MetricGroup]):
    container_set = set()
    for m in metrics:
        for c in m.requires():
            container_set.add(c)
    container_list = [c() for c in container_set]
    return container_list
