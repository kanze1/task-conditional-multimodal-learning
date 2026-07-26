from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tcmi.config import output_root
from tcmi.evaluation.probe import run_probes
from tcmi.training.trainer import train_run


def matrix_cells(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    matrix = config["matrix"]
    for architecture in matrix["architectures"]:
        for train_mode in matrix["train_modes"]:
            for condition in matrix["conditions"]:
                if train_mode == "multimodal_conflict" and condition != "conflict":
                    continue
                if train_mode == "multimodal_aligned" and condition == "conflict":
                    continue
                for seed in matrix["seeds"]:
                    yield {
                        "architecture": architecture,
                        "train_mode": train_mode,
                        "condition": condition,
                        "seed": int(seed),
                    }


def run_dir_for(config: dict[str, Any], cell: dict[str, Any]) -> Path:
    evidence = config["project"]["evidence_level"]
    slug = (
        f"{evidence}__{cell['architecture']}__{cell['train_mode']}"
        f"__{cell['condition']}__seed_{cell['seed']}"
    )
    return output_root(config) / "runs" / slug


def command_plan(config_path: str | Path, config: dict[str, Any]) -> list[str]:
    quoted_config = _powershell_quote(str(config_path))
    commands = [
        f"python -m tcmi generate --config {quoted_config}",
        f"python -m tcmi audit --config {quoted_config}",
    ]
    for cell in matrix_cells(config):
        commands.append(
            "python -m tcmi train "
            f"--config {quoted_config} "
            f"--architecture {cell['architecture']} "
            f"--train-mode {cell['train_mode']} "
            f"--condition {cell['condition']} "
            f"--seed {cell['seed']}"
        )
    for cell in matrix_cells(config):
        run_dir = _powershell_quote(str(run_dir_for(config, cell)))
        for scope in config["matrix"]["representation_scopes"]:
            commands.append(
                f"python -m tcmi probe --config {quoted_config} "
                f"--run-dir {run_dir} --representation-scope {scope}"
            )
    commands.extend(
        [
            f"python -m tcmi aggregate --config {quoted_config}",
            f"python -m tcmi decide --config {quoted_config}",
        ]
    )
    return commands


def execute_matrix(
    config: dict[str, Any],
    *,
    stage: str,
) -> list[str]:
    completed: list[str] = []
    if stage in {"train", "all"}:
        for cell in matrix_cells(config):
            path = train_run(config=config, **cell)
            completed.append(str(path))
    if stage in {"probe", "all"}:
        for cell in matrix_cells(config):
            run_dir = run_dir_for(config, cell)
            for scope in config["matrix"]["representation_scopes"]:
                path = run_probes(config, run_dir, scope)
                completed.append(str(path))
    return completed


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
