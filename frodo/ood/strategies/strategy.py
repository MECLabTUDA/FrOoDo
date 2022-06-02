from torch.utils.data import DataLoader

from ...data.utils import sample_collate


class OODStrategy:
    def get_dataloader(self, **dataloader_kwargs):
        return DataLoader(self.dataset, collate_fn=sample_collate, **dataloader_kwargs)
