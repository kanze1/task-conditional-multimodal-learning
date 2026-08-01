from __future__ import annotations

import torch

from tcmi.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_carries_required_provenance(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        config_hash="config-hash",
        dataset_manifest_hash="dataset-hash",
        run_identity={
            "architecture": "tiny_cnn",
            "train_mode": "image_only",
            "condition": "redundant",
            "seed": 2601,
        },
        budget={"optimizer_updates": 7, "gpu_time_seconds": 1.25},
    )
    payload = load_checkpoint(path, torch.device("cpu"))
    assert payload["schema_version"] == "tcmi_checkpoint_v1"
    assert payload["config_hash"] == "config-hash"
    assert payload["dataset_manifest_hash"] == "dataset-hash"
    assert payload["run_identity"]["seed"] == 2601
    assert payload["budget"]["optimizer_updates"] == 7
    assert payload["budget"]["gpu_time_seconds"] == 1.25
