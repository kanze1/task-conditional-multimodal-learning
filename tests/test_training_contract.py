from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tcmi.evidence as evidence_module
from tcmi.config import public_config, stable_hash
from tcmi.data.audit import audit_dataset
from tcmi.data.generator import generate_dataset
from tcmi.io import read_json
from tcmi.models import build_model
from tcmi.training.checkpoint import load_checkpoint
from tcmi.training.losses import symmetric_contrastive_loss
from tcmi.training.trainer import (
    TrainingError,
    _configure_trainable_parameters,
    _mode_loss,
    _validate_identity,
    train_run,
)


def test_contrastive_loss_requires_paired_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        symmetric_contrastive_loss(torch.rand(3, 4), torch.rand(2, 4), 0.07)


@pytest.mark.parametrize(
    ("mode", "expected_images", "expected_texts"),
    [
        ("image_only", 2, 0),
        ("image_only_matched", 1, 1),
        ("text_only", 0, 2),
        ("multimodal_aligned", 1, 1),
        ("multimodal_shuffled", 1, 1),
        ("multimodal_conflict", 1, 1),
    ],
)
def test_training_modes_report_encoding_budget(
    smoke_config: dict,
    mode: str,
    expected_images: int,
    expected_texts: int,
) -> None:
    model = build_model(smoke_config, "tiny_cnn")
    images = torch.rand(4, 3, 32, 32)
    input_ids = torch.zeros(4, 14, dtype=torch.long)
    attention_mask = torch.ones(4, 14, dtype=torch.bool)
    loss, image_count, text_count = _mode_loss(
        model=model,
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        train_mode=mode,
        image_noise_std=0.01,
        text_dropout_probability=0.1,
    )
    assert loss.ndim == 0
    assert image_count == expected_images
    assert text_count == expected_texts


def test_strict_image_only_freezes_text_tower(smoke_config: dict) -> None:
    model = build_model(smoke_config, "tiny_cnn")
    _configure_trainable_parameters(model, "image_only")
    assert all(not parameter.requires_grad for parameter in model.text_encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.image_encoder.parameters())


def test_conflict_mode_requires_conflict_condition() -> None:
    with pytest.raises(TrainingError, match="必须使用 conflict"):
        _validate_identity("tiny_cnn", "multimodal_conflict", "redundant")


def test_conflict_condition_uses_single_canonical_multimodal_mode() -> None:
    with pytest.raises(TrainingError, match="避免重复"):
        _validate_identity("tiny_cnn", "multimodal_aligned", "conflict")


def test_final_checkpoint_budget_matches_final_budget(
    tiny_temp_config: dict,
    monkeypatch,
) -> None:
    config = tiny_temp_config
    config["training"]["device"] = "cpu"
    config["training"]["batch_size"] = 4
    config["training"]["epochs"] = 1
    config["_meta"]["config_hash"] = stable_hash(public_config(config))
    project_root = Path(config["_meta"]["config_path"]).parent.parent
    protocol_path = project_root / config["project"]["protocol_path"]
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("draft protocol\n", encoding="utf-8")
    monkeypatch.setattr(
        evidence_module,
        "source_snapshot",
        lambda root: {
            "git_commit": "test-commit",
            "dirty": False,
            "dirty_paths": [],
        },
    )

    generate_dataset(config)
    audit_dataset(config)
    run_dir = train_run(
        config,
        architecture="tiny_cnn",
        train_mode="image_only",
        condition="redundant",
        seed=2601,
    )
    manifest = read_json(run_dir / "run_manifest.json")
    budget = read_json(run_dir / "budget.json")
    checkpoint = load_checkpoint(
        run_dir / manifest["final_checkpoint"],
        torch.device("cpu"),
    )
    assert checkpoint["budget"] == budget
    assert budget["wall_clock_seconds"] > 0
