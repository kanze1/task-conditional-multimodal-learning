from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from tcmi.config import public_config, stable_hash
from tcmi.constants import REPRESENTATION_SCOPES
from tcmi.data.dataset import ShardedSceneGraphDataset, dataset_manifest_hash
from tcmi.io import read_json, sha256_file
from tcmi.models import build_model
from tcmi.training.checkpoint import load_checkpoint
from tcmi.training.trainer import resolve_device


@dataclass
class FeatureSet:
    features: torch.Tensor
    labels: torch.Tensor
    unseen_mask: torch.Tensor
    sample_ids: list[str]


def load_model_for_run(
    config: dict[str, Any],
    run_dir: str | Path,
) -> tuple[Any, torch.device, dict[str, Any], Path]:
    directory = Path(run_dir)
    run_manifest = read_json(directory / "run_manifest.json")
    if run_manifest.get("status") != "completed":
        raise RuntimeError("只有 completed 训练运行可以执行 probe")
    if run_manifest["config_hash"] != stable_hash(public_config(config)):
        raise RuntimeError("当前配置与训练运行的 config_hash 不一致")
    if run_manifest["dataset_manifest_hash"] != dataset_manifest_hash(config):
        raise RuntimeError("当前数据 manifest 与训练运行不一致")
    identity = run_manifest["identity"]
    device = resolve_device(str(config["training"]["device"]))
    model = build_model(config, identity["architecture"]).to(device)
    checkpoint_path = directory / run_manifest["final_checkpoint"]
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint["config_hash"] != config["_meta"]["config_hash"]:
        raise RuntimeError("checkpoint 的 config_hash 不匹配")
    if checkpoint["dataset_manifest_hash"] != dataset_manifest_hash(config):
        raise RuntimeError("checkpoint 的 dataset_manifest_hash 不匹配")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, device, run_manifest, checkpoint_path


@torch.inference_mode()
def extract_features(
    *,
    config: dict[str, Any],
    model: Any,
    device: torch.device,
    condition: str,
    split: str,
    representation_scope: str,
    corruption: str = "clean",
    seed: int = 0,
) -> FeatureSet:
    if representation_scope not in REPRESENTATION_SCOPES:
        raise ValueError(f"未知 representation_scope: {representation_scope}")
    base_dataset = ShardedSceneGraphDataset(config, condition, split)
    conflict_dataset = (
        ShardedSceneGraphDataset(config, "conflict", split)
        if corruption == "conflicting_text"
        else None
    )
    loader = DataLoader(
        base_dataset,
        batch_size=int(config["probe"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        persistent_workers=int(config["training"]["num_workers"]) > 0,
    )
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    feature_batches = []
    label_batches = []
    unseen_batches = []
    sample_ids: list[str] = []
    offset = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        batch_size = images.shape[0]

        if corruption == "noisy_image":
            noise = torch.randn(
                images.shape,
                generator=generator,
                device=device,
                dtype=images.dtype,
            )
            images = (images + 0.15 * noise).clamp(0, 1)
        elif corruption == "noisy_text":
            input_ids = _mask_text_values(input_ids, generator, probability=0.5)
        elif corruption == "conflicting_text":
            if conflict_dataset is None:
                raise RuntimeError("conflict_dataset 未初始化")
            conflict_rows = [
                conflict_dataset[index] for index in range(offset, offset + batch_size)
            ]
            conflict_ids = [row["sample_id"] for row in conflict_rows]
            if conflict_ids != list(batch["sample_id"]):
                raise RuntimeError("冲突文本与基础样本未按 sample_id 对齐")
            input_ids = torch.stack([row["input_ids"] for row in conflict_rows]).to(device)
            attention_mask = torch.stack(
                [row["attention_mask"] for row in conflict_rows]
            ).to(device)

        image_features = None
        text_features = None
        if representation_scope in {"image", "fused"}:
            image_features = model.encode_image(images)
        if representation_scope in {"text", "fused"}:
            text_features = model.encode_text(input_ids, attention_mask)

        if corruption == "missing_image":
            if representation_scope != "fused":
                raise ValueError("missing_image 只适用于 fused representation")
            image_features = torch.zeros_like(image_features)
        if corruption == "missing_text":
            if representation_scope != "fused":
                raise ValueError("missing_text 只适用于 fused representation")
            text_features = torch.zeros_like(text_features)

        if representation_scope == "image":
            features = image_features
        elif representation_scope == "text":
            features = text_features
        else:
            features = torch.cat((image_features, text_features), dim=-1)
        if features is None:
            raise RuntimeError("feature extraction 未产生特征")

        feature_batches.append(features.cpu())
        label_batches.append(batch["labels"].cpu())
        unseen_batches.append(batch["is_unseen_combination"].bool().cpu())
        sample_ids.extend(list(batch["sample_id"]))
        offset += batch_size

    return FeatureSet(
        features=torch.cat(feature_batches),
        labels=torch.cat(label_batches),
        unseen_mask=torch.cat(unseen_batches),
        sample_ids=sample_ids,
    )


def _mask_text_values(
    input_ids: torch.Tensor,
    generator: torch.Generator,
    probability: float,
) -> torch.Tensor:
    from tcmi.data.vocab import VOCABULARY

    masked = input_ids.clone()
    positions = torch.tensor([2, 4, 6, 8, 10, 12], device=input_ids.device)
    selection = torch.rand(
        (input_ids.shape[0], len(positions)),
        generator=generator,
        device=input_ids.device,
    ) < probability
    values = masked[:, positions]
    values[selection] = VOCABULARY.mask_id
    masked[:, positions] = values
    return masked


def checkpoint_provenance(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path)
    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
    }
