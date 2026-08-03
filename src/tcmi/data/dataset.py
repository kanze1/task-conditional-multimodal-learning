from __future__ import annotations

from bisect import bisect_right
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from tcmi.data.generator import dataset_root
from tcmi.io import read_json


class ShardedSceneGraphDataset(Dataset[dict[str, Any]]):
    def __init__(self, config: dict[str, Any], condition: str, split: str) -> None:
        self.root = dataset_root(config)
        self.condition = condition
        self.split = split
        manifest_path = self.root / condition / "manifest.json"
        manifest = read_json(manifest_path)
        split_manifest = manifest["splits"][split]
        self.shards = split_manifest["shards"]
        self.length = int(split_manifest["count"])
        self.cumulative_counts: list[int] = []
        running = 0
        for shard in self.shards:
            running += int(shard["count"])
            self.cumulative_counts.append(running)
        if running != self.length:
            raise ValueError("manifest 中的 shard 计数与 split 总数不一致")
        self._cached_shard_index: int | None = None
        self._cached_arrays: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative_counts, index)
        previous = 0 if shard_index == 0 else self.cumulative_counts[shard_index - 1]
        local_index = index - previous
        arrays = self._load_shard(shard_index)
        return {
            "image": torch.from_numpy(arrays["images"][local_index].copy()).float() / 255.0,
            "input_ids": torch.from_numpy(arrays["input_ids"][local_index].copy()).long(),
            "attention_mask": torch.from_numpy(
                arrays["attention_mask"][local_index].copy()
            ).bool(),
            "labels": torch.from_numpy(arrays["labels"][local_index].copy()).long(),
            "is_unseen_combination": bool(
                arrays["is_unseen_combination"][local_index]
            ),
            "sample_id": str(arrays["sample_ids"][local_index]),
        }

    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        if self._cached_shard_index != shard_index:
            shard_path = self.root / self.shards[shard_index]["path"]
            with np.load(shard_path, allow_pickle=False) as archive:
                self._cached_arrays = {key: archive[key] for key in archive.files}
            self._cached_shard_index = shard_index
        if self._cached_arrays is None:
            raise RuntimeError("shard cache 未初始化")
        return self._cached_arrays


class ResidentSceneGraphData:
    """把一个 (condition, split) 的全部 shard 一次性驻留为设备张量。

    图像保持 uint8 存储，仅在取 batch 时转换为 float，训练全程不再经过
    DataLoader worker 与逐样本 collate。
    """

    def __init__(
        self,
        config: dict[str, Any],
        condition: str,
        split: str,
        device: torch.device,
    ) -> None:
        source = ShardedSceneGraphDataset(config, condition, split)
        images: list[np.ndarray] = []
        input_ids: list[np.ndarray] = []
        attention_mask: list[np.ndarray] = []
        for shard in source.shards:
            shard_path = source.root / shard["path"]
            with np.load(shard_path, allow_pickle=False) as archive:
                images.append(archive["images"])
                input_ids.append(archive["input_ids"])
                attention_mask.append(archive["attention_mask"])
        self.device = device
        self.images = torch.from_numpy(np.concatenate(images)).to(device)
        self.input_ids = torch.from_numpy(np.concatenate(input_ids)).long().to(device)
        self.attention_mask = (
            torch.from_numpy(np.concatenate(attention_mask)).bool().to(device)
        )
        self.length = int(self.images.shape[0])
        if self.length != len(source):
            raise ValueError("驻留样本数与 manifest split 计数不一致")

    def __len__(self) -> int:
        return self.length

    def training_batches(
        self,
        batch_size: int,
        generator: torch.Generator,
    ):
        permutation = torch.randperm(self.length, generator=generator).to(self.device)
        for start in range(0, self.length - batch_size + 1, batch_size):
            index = permutation[start : start + batch_size]
            yield {
                "image": self.images.index_select(0, index).float().div_(255.0),
                "input_ids": self.input_ids.index_select(0, index),
                "attention_mask": self.attention_mask.index_select(0, index),
            }


def dataset_manifest_hash(config: dict[str, Any]) -> str:
    from tcmi.io import sha256_file

    return sha256_file(dataset_root(config) / "manifest.json")
