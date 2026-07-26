from __future__ import annotations

import pytest
import torch

from tcmi.models import build_model, parameter_audit


@pytest.mark.parametrize("architecture", ["tiny_vit", "tiny_cnn"])
def test_dual_tower_output_contract(smoke_config: dict, architecture: str) -> None:
    model = build_model(smoke_config, architecture)
    images = torch.rand(3, 3, 32, 32)
    input_ids = torch.zeros(3, 14, dtype=torch.long)
    attention_mask = torch.ones(3, 14, dtype=torch.bool)
    image_features = model.encode_image(images)
    text_features = model.encode_text(input_ids, attention_mask)
    assert image_features.shape == (3, 64)
    assert text_features.shape == (3, 64)
    assert torch.allclose(image_features.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(text_features.norm(dim=-1), torch.ones(3), atol=1e-5)
    audit = parameter_audit(model)
    assert audit["total_parameters"] > 0
    assert audit["image_encoder_parameters"] > 0
    assert audit["text_encoder_parameters"] > 0
