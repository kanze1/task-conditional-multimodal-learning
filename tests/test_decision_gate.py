from __future__ import annotations

from copy import deepcopy

import pytest

from tcmi.config import ConfigError, validate_config
from tcmi.evaluation.decision import _evaluate_formal_thresholds


def test_formal_evidence_cannot_be_enabled_by_label_only(smoke_config: dict) -> None:
    config = deepcopy(smoke_config)
    config["project"]["evidence_level"] = "formal"
    config["project"]["protocol_status"] = "draft"
    with pytest.raises(ConfigError):
        validate_config(config)


def test_formal_complementary_gate_counts_only_preregistered_fused_tasks() -> None:
    rows = []
    for architecture in ("tiny_cnn", "tiny_vit"):
        for task in ("entity", "joint_graph"):
            rows.append(
                {
                    "architecture": architecture,
                    "train_mode": "multimodal_aligned",
                    "condition": "complementary",
                    "representation_scope": "fused",
                    "task": task,
                    "metric": "clean_accuracy",
                    "mean_delta": "0.10",
                    "sample_std": "0.01",
                }
            )
        rows.append(
            {
                "architecture": architecture,
                "train_mode": "multimodal_aligned",
                "condition": "complementary",
                "representation_scope": "image",
                "task": "predicate",
                "metric": "clean_accuracy",
                "mean_delta": "0.50",
                "sample_std": "0.01",
            }
        )

    decision = _evaluate_formal_thresholds(
        rows,
        {"matrix_completeness": {"complete": True}},
    )

    assert decision["task_passes"] == {"entity": True, "joint_graph": True}
    assert decision["checks"]["complementary_two_tasks"] is True
