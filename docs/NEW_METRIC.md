
# Add new MetricGroup

```python
class MetricGroup:
    def __init__(self) -> None:
        pass

    def requires(self) -> List[Container]:
        raise NotImplementedError("Please Implement this method")

    def __call__(self, data_container: List[Container], experiment_metadata: dict):
        raise NotImplementedError("Please Implement this method")

    def present(self, **kwargs):
        pass
```
