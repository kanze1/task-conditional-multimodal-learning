from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config_hash: str,
    dataset_manifest_hash: str,
    run_identity: dict[str, Any],
    budget: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema_version": "tcmi_checkpoint_v1",
        "epoch": epoch,
        "config_hash": config_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "run_identity": run_identity,
        "budget": budget,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)
