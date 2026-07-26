from __future__ import annotations

from copy import deepcopy

import pytest

from tcmi.config import ConfigError, canonical_json, stable_hash, validate_config
from tcmi.evidence import EvidenceError, enforce_evidence_gate


def test_stable_hash_ignores_mapping_insertion_order() -> None:
    left = {"alpha": 1, "beta": {"x": 2, "y": 3}}
    right = {"beta": {"y": 3, "x": 2}, "alpha": 1}
    assert canonical_json(left) == canonical_json(right)
    assert stable_hash(left) == stable_hash(right)


def test_formal_config_requires_frozen_protocol(smoke_config: dict) -> None:
    config = deepcopy(smoke_config)
    config["project"]["evidence_level"] = "formal"
    with pytest.raises(ConfigError, match="协议未冻结"):
        validate_config(config)


def test_cli_cannot_upgrade_smoke_to_formal(smoke_config: dict) -> None:
    with pytest.raises(EvidenceError, match="禁止"):
        enforce_evidence_gate(smoke_config, requested_level="formal")


def test_all_four_information_conditions_are_required(smoke_config: dict) -> None:
    config = deepcopy(smoke_config)
    config["data"]["conditions"] = ["redundant", "complementary"]
    with pytest.raises(ConfigError, match="四种"):
        validate_config(config)


def test_matched_control_requires_even_batch(smoke_config: dict) -> None:
    config = deepcopy(smoke_config)
    config["training"]["batch_size"] = 31
    with pytest.raises(ConfigError, match="偶数"):
        validate_config(config)
