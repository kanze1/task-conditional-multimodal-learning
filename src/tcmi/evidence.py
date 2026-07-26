from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tcmi.config import output_root, project_path, public_config, stable_hash
from tcmi.io import append_jsonl, sha256_file, write_json


class EvidenceError(RuntimeError):
    """Raised when evidence provenance would be ambiguous or overstated."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def enforce_evidence_gate(config: dict[str, Any], requested_level: str | None = None) -> str:
    configured_level = str(config["project"]["evidence_level"])
    evidence_level = requested_level or configured_level
    if evidence_level == "formal" and config["project"].get("protocol_status") != "frozen":
        raise EvidenceError("协议状态不是 frozen，禁止创建 formal 结果")
    if evidence_level == "formal" and configured_level != "formal":
        raise EvidenceError("禁止通过命令行把 smoke/pilot 配置升级为 formal")
    return evidence_level


@dataclass(frozen=True)
class RunIdentity:
    architecture: str
    train_mode: str
    condition: str
    seed: int
    evidence_level: str

    @property
    def slug(self) -> str:
        return (
            f"{self.evidence_level}__{self.architecture}__{self.train_mode}"
            f"__{self.condition}__seed_{self.seed}"
        )


class RunContext:
    def __init__(
        self,
        config: dict[str, Any],
        identity: RunIdentity,
        dataset_manifest_hash: str,
    ) -> None:
        self.config = config
        self.identity = identity
        self.dataset_manifest_hash = dataset_manifest_hash
        self.run_dir = output_root(config) / "runs" / identity.slug

        resolved_config = public_config(config)
        source = _source_snapshot(project_path(config))
        if identity.evidence_level == "formal" and source["dirty"]:
            raise EvidenceError("formal 运行要求 Git 工作树干净")
        protocol_path = project_path(config, config["project"]["protocol_path"])
        if not protocol_path.is_file():
            raise EvidenceError(f"找不到协议文件: {protocol_path}")
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.log_path = self.run_dir / "events.jsonl"
        run_manifest = {
            "schema_version": "tcmi_run_v1",
            "created_at": utc_now(),
            "status": "created",
            "evidence_level": identity.evidence_level,
            "identity": {
                "architecture": identity.architecture,
                "train_mode": identity.train_mode,
                "condition": identity.condition,
                "seed": identity.seed,
            },
            "config_hash": stable_hash(resolved_config),
            "dataset_manifest_hash": dataset_manifest_hash,
            "protocol_path": str(protocol_path.relative_to(project_path(config))),
            "protocol_sha256": sha256_file(protocol_path),
            "source": source,
            "environment": _environment_snapshot(),
            "python_version": sys.version,
            "platform": platform.platform(),
        }
        write_json(self.run_dir / "resolved_config.json", resolved_config)
        write_json(self.run_dir / "run_manifest.json", run_manifest)

    def log(self, event: str, **payload: Any) -> None:
        append_jsonl(
            self.log_path,
            {
                "timestamp": utc_now(),
                "event": event,
                **payload,
            },
        )

    def update_status(self, status: str, **payload: Any) -> None:
        manifest_path = self.run_dir / "run_manifest.json"
        from tcmi.io import read_json

        manifest = read_json(manifest_path)
        manifest["status"] = status
        manifest["updated_at"] = utc_now()
        manifest.update(payload)
        write_json(manifest_path, manifest)


def _source_snapshot(project_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    return {
        "git_commit": commit,
        "dirty": bool(status),
        "dirty_paths": status,
    }


def _environment_snapshot() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    import torch

    packages = {}
    for name in ("numpy", "pyyaml", "matplotlib", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
