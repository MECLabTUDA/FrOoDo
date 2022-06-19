
# Add new Method

A new Method for OOD Detection needs to be an instance of Class OODMethod. As shown below the only methods that must be specified is the calculate_ood_score method which is returs a batch of scores (classification) or masks of scores (Segmentation) for a Sample Batch.
```python
class OODMethod:
    def __init__(self, hyperparams={}) -> None:
        pass

    def get_params(self, dict=False):
        return "no params" if not dict else {}

    def display_name(self) -> str:
        return f"{type(self).__name__}"

    def modify_net(self, net):
        return net

    def remodify_net(self, net):
        return net

    def calculate_ood_score(self, imgs, net, batch=None):
        raise NotImplementedError("Please Implement this method")

    def _set_metadata(self, batch: Batch, scores) -> Batch:
        batch["metadata"].append_to_keyed_dict(
            SampleMetadataCommonTypes.OOD_SCORE.name,
            self.display_name(),
            scores,
        )
        return batch

    def __call__(
        self,
        batch: Batch,
        net,
        task_type: TaskType = TaskType.SEGMENTATION,
        remove_ignore_index: bool = True,
        score_reduction_method: Callable = np.mean,
    ) -> Sample:
        # modify network if needed
        net = self.modify_net(net)

        scores = self.calculate_ood_score(batch.image, net, batch).numpy()

        ##########################
        # POSTPROCESSING removed #
        ##########################

        # save scores in metadata to later be accessed by the metrics
        batch = self._set_metadata(batch, scores)

        # always remodify network so it works properly for other mothods
        self.remodify_net(net)

        return batch


```