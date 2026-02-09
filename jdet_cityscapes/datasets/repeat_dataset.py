import numpy as np
from jittor.dataset import Dataset

from jdet.utils.registry import DATASETS, build_from_cfg


@DATASETS.register_module()
class RepeatDataset(Dataset):
    """Repeat a dataset N times (MMDetection-style)."""

    def __init__(self, dataset, times, **kwargs):
        if isinstance(dataset, dict):
            dataset = build_from_cfg(dataset, DATASETS)
        self.dataset = dataset
        # print(f"[dbg] RepeatDataset: inner dataset type={type(self.dataset)}", flush=True)
        self.times = int(times)
        self.CLASSES = getattr(dataset, "CLASSES", None)
        if hasattr(self.dataset, "flag"):
            self.flag = np.tile(self.dataset.flag, self.times)

        base_len = getattr(self.dataset, "total_len", None)
        if base_len is None and hasattr(self.dataset, "img_ids"):
            base_len = len(self.dataset.img_ids)
        if base_len is None and hasattr(self.dataset, "coco"):
            base_len = len(self.dataset.coco.getImgIds())
        if base_len is None:
            base_len = len(self.dataset)
        self._ori_len = int(base_len)

        batch_size = getattr(dataset, "batch_size", 1)
        num_workers = getattr(dataset, "num_workers", 0)
        shuffle = getattr(dataset, "shuffle", False)
        drop_last = getattr(dataset, "drop_last", False)
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.total_len = self.times * self._ori_len

    def __getitem__(self, idx):
        # print(f"[dbg] RepeatDataset.__getitem__: {idx}", flush=True)
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len

    def collate_batch(self, batch):
        if hasattr(self.dataset, "collate_batch"):
            return self.dataset.collate_batch(batch)
        return super().collate_batch(batch)
