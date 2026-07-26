from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tcmi.config import load_config


@pytest.fixture
def smoke_config() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "mvp_smoke.yaml")


@pytest.fixture
def tiny_temp_config(
    smoke_config: dict[str, Any],
    tmp_path: Path,
) -> dict[str, Any]:
    config = deepcopy(smoke_config)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config["_meta"]["config_path"] = str(config_dir / "test.yaml")
    config["data"]["splits"] = {
        "train": 16,
        "validation": 8,
        "test": 8,
    }
    config["data"]["shard_size"] = 4
    config["data"]["preview_samples_per_split"] = 2
    return config
