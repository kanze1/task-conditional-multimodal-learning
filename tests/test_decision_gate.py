from __future__ import annotations

from copy import deepcopy

import pytest

from tcmi.config import ConfigError, validate_config


def test_formal_evidence_cannot_be_enabled_by_label_only(smoke_config: dict) -> None:
    config = deepcopy(smoke_config)
    config["project"]["evidence_level"] = "formal"
    config["project"]["protocol_status"] = "draft"
    with pytest.raises(ConfigError):
        validate_config(config)
