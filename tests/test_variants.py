from __future__ import annotations

from copy import deepcopy

import pytest

from tcmi.config import ConfigError, validate_config
from tcmi.data.audit import audit_dataset
from tcmi.data.generator import generate_dataset, render_image, sample_graph
from tcmi.data.oracle import _chance_ceiling, _partial_information_ceiling


def test_key_correlation_default_is_bit_identical() -> None:
    for index in range(20):
        baseline = sample_graph(20260726, "train", index, heldout_modulus=4)
        explicit = sample_graph(
            20260726, "train", index, heldout_modulus=4, key_correlation=0.5
        )
        assert baseline == explicit


def test_key_correlation_skews_agreement_rate() -> None:
    graphs = [
        sample_graph(
            20260726, "train", index, heldout_modulus=4, key_correlation=0.9
        )
        for index in range(500)
    ]
    agreement = sum(g.text_key == g.visual_key for g in graphs) / len(graphs)
    assert 0.85 <= agreement <= 0.95
    baseline = [
        sample_graph(20260726, "train", index, heldout_modulus=4)
        for index in range(500)
    ]
    for varied, original in zip(graphs, baseline, strict=True):
        assert varied.visual_key == original.visual_key
        assert varied.subject_shape == original.subject_shape
        assert varied.scene_nonce == original.scene_nonce


def test_background_salience_encodes_visual_key() -> None:
    for index in range(30):
        graph = sample_graph(20260726, "train", index, heldout_modulus=4)
        default_image = render_image(graph, "complementary", 32)
        salient_image = render_image(
            graph, "complementary", 32, visual_key_salience="background"
        )
        expected_background = 64 if graph.visual_key else 16
        assert salient_image[0, 0, 16] == expected_background
        assert default_image[0, 0, 16] == 24
        # 前景内容（形状、关系、角落 patch）保持一致
        foreground = default_image != 24
        assert (salient_image[foreground] == default_image[foreground]).all()


def test_oracle_ceilings_parameterized_by_correlation() -> None:
    assert _partial_information_ceiling(
        "joint_graph", 5000
    ) == _partial_information_ceiling("joint_graph", 5000, 0.5)
    assert _partial_information_ceiling("joint_graph", 5000, 0.9) > 0.9
    assert _chance_ceiling("joint_graph", 5000) == pytest.approx(
        0.25 + 3 * (0.25 * 0.75 / 5000) ** 0.5
    )
    assert _chance_ceiling("joint_graph", 5000, 0.9) == pytest.approx(
        0.45 + 3 * (0.45 * 0.55 / 5000) ** 0.5
    )
    assert _chance_ceiling("entity", 5000, 0.9) == _chance_ceiling("entity", 5000)


def test_config_rejects_invalid_variant_fields(smoke_config: dict) -> None:
    config = deepcopy(smoke_config)
    config["data"]["key_correlation"] = 1.2
    with pytest.raises(ConfigError, match="key_correlation"):
        validate_config(config)
    config = deepcopy(smoke_config)
    config["data"]["visual_key_salience"] = "huge"
    with pytest.raises(ConfigError, match="visual_key_salience"):
        validate_config(config)


def test_variant_dataset_generation_and_audit_pass(tiny_temp_config: dict) -> None:
    config = tiny_temp_config
    config["data"]["visual_key_salience"] = "background"
    config["data"]["key_correlation"] = 0.9
    generate_dataset(config)
    report_path = audit_dataset(config)
    assert report_path.is_file()
