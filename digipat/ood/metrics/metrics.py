class MetricGroup:
    def __init__(self) -> None:
        pass

    def requieres(self):
        raise NotImplementedError("Please Implement this method")

    def __call__(self, data_container):
        raise NotImplementedError("Please Implement this method")

    def present(self):
        pass


class Metric:
    def __init__(self) -> None:
        pass

    def __call__(self, data_container):
        raise NotImplementedError("Please Implement this method")

    def present(self):
        pass
