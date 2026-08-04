from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tcmi.constants import (
    ARCHITECTURES,
    EVIDENCE_LEVELS,
    INFORMATION_CONDITIONS,
    REPRESENTATION_SCOPES,
    SPLITS,
    TRAIN_MODES,
)


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the MVP contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError("配置文件顶层必须是 mapping")
    config = deepcopy(loaded)
    config["_meta"] = {
        "config_path": str(config_path),
        "config_hash": stable_hash(loaded),
    }
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = ("project", "data", "model", "training", "probe", "matrix")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ConfigError(f"缺少配置段: {missing}")

    evidence_level = config["project"].get("evidence_level")
    if evidence_level not in EVIDENCE_LEVELS:
        raise ConfigError(f"未知 evidence_level: {evidence_level}")
    if evidence_level == "formal" and config["project"].get("protocol_status") != "frozen":
        raise ConfigError("协议未冻结时禁止生成 formal 证据")
    if float(config["project"].get("gpu_hour_budget", 0)) <= 0:
        raise ConfigError("gpu_hour_budget 必须为正数")
    if Path(config["project"].get("protocol_path", "")).is_absolute():
        raise ConfigError("protocol_path 必须是相对项目根目录的路径")

    data = config["data"]
    conditions = tuple(data.get("conditions", ()))
    if set(conditions) != set(INFORMATION_CONDITIONS):
        raise ConfigError("MVP 数据必须完整包含四种预注册信息条件")
    if data.get("image_size", 0) < 16:
        raise ConfigError("image_size 必须至少为 16")
    if data.get("schema_version") != "tcmi_scene_graph_v2":
        raise ConfigError("MVP 数据 schema_version 必须为 tcmi_scene_graph_v2")
    if data.get("max_seq_length") != 14:
        raise ConfigError("tcmi_scene_graph_v2 的 max_seq_length 必须为 14")
    if any(int(size) <= 0 for size in data.get("splits", {}).values()):
        raise ConfigError("所有 split 大小必须为正整数")
    if set(data.get("splits", {})) != set(SPLITS):
        raise ConfigError("data.splits 必须恰好包含 train/validation/test")
    if int(data.get("shard_size", 0)) <= 0:
        raise ConfigError("shard_size 必须为正整数")
    if int(data.get("heldout_modulus", 0)) <= 1:
        raise ConfigError("heldout_modulus 必须大于 1")
    key_correlation = float(data.get("key_correlation", 0.5))
    if not 0.5 <= key_correlation <= 1.0:
        raise ConfigError("key_correlation 必须位于 [0.5, 1.0]")
    salience = data.get("visual_key_salience", "patch")
    if salience not in {"patch", "background"}:
        raise ConfigError(f"未知 visual_key_salience: {salience}")

    matrix = config["matrix"]
    _validate_subset(matrix.get("architectures", ()), ARCHITECTURES, "architecture")
    _validate_subset(matrix.get("train_modes", ()), TRAIN_MODES, "train_mode")
    _validate_subset(matrix.get("conditions", ()), INFORMATION_CONDITIONS, "condition")
    _validate_subset(
        matrix.get("representation_scopes", ()),
        REPRESENTATION_SCOPES,
        "representation_scope",
    )
    seeds = matrix.get("seeds", ())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ConfigError("matrix.seeds 必须为非空且不重复")
    if len(matrix["architectures"]) < 2:
        raise ConfigError("MVP 矩阵必须包含至少两种 architecture")
    if evidence_level in {"pilot", "formal"} and len(seeds) < 3:
        raise ConfigError("pilot/formal 矩阵必须至少包含三个 seed")

    batch_size = int(config["training"].get("batch_size", 0))
    if batch_size < 4 or batch_size % 2:
        raise ConfigError("training.batch_size 必须为偶数且至少为 4")
    precision = config["training"].get("precision", "fp32")
    if precision not in {"fp32", "bf16"}:
        raise ConfigError(f"未知 training.precision: {precision}")
    data_backend = config["training"].get("data_backend", "dataloader")
    if data_backend not in {"dataloader", "gpu_resident"}:
        raise ConfigError(f"未知 training.data_backend: {data_backend}")
    if not isinstance(config["training"].get("tf32", False), bool):
        raise ConfigError("training.tf32 必须为布尔值")
    heads = int(config["model"].get("attention_heads", 0))
    for field in ("image_width", "text_width"):
        width = int(config["model"].get(field, 0))
        if heads <= 0 or width % heads:
            raise ConfigError(f"{field} 必须能被 attention_heads 整除")

    confidence = float(config["probe"].get("confidence_level", 0))
    if not 0 < confidence < 1:
        raise ConfigError("probe.confidence_level 必须位于 (0, 1)")


def _validate_subset(values: Any, allowed: tuple[str, ...], field_name: str) -> None:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ConfigError(f"未知 {field_name}: {unknown}")


def project_path(config: dict[str, Any], *parts: str) -> Path:
    config_path = Path(config["_meta"]["config_path"])
    root = config_path.parent.parent
    return root.joinpath(*parts).resolve()


def output_root(config: dict[str, Any]) -> Path:
    configured = Path(config["project"]["output_root"])
    if configured.is_absolute():
        raise ConfigError("output_root 必须是相对项目根目录的路径")
    return project_path(config, str(configured))


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "_meta"}
