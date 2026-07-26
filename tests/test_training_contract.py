from __future__ import annotations

import pytest
import torch

from tcmi.models import build_model
from tcmi.training.losses import symmetric_contrastive_loss
from tcmi.training.trainer import (
    TrainingError,
    _configure_trainable_parameters,
    _mode_loss,
    _validate_identity,
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
