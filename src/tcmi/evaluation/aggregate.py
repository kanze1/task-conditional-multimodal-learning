from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from tcmi.config import output_root
from tcmi.io import read_json, write_json

GROUP_FIELDS = (
    "architecture",
    "train_mode",
    "condition",
    "representation_scope",
    "task",
    "metric",
)


class AggregationError(RuntimeError):
    """Raised when result provenance is mixed or incomplete."""


def aggregate_results(config: dict[str, Any]) -> Path:
    evidence_level = config["project"]["evidence_level"]
    result_files = sorted(
        output_root(config).glob("runs/*/probes/*/probe_metrics.json")
    )
    records: list[dict[str, Any]] = []
    for path in result_files:
        payload = read_json(path)
        for metric in payload["metrics"]:
            if metric["evidence_level"] != evidence_level:
                continue
            records.append({**metric, "source_path": str(path)})
    if not records:
        raise AggregationError(f"没有可聚合的 {evidence_level} probe 结果")

    output_dir = output_root(config) / "aggregates" / evidence_level
    if output_dir.exists():
        raise AggregationError(f"聚合目录已存在，拒绝覆盖: {output_dir}")
    output_dir.mkdir(parents=True)

    seed_rows = sorted(
        records,
        key=lambda row: tuple(str(row[field]) for field in GROUP_FIELDS)
        + (int(row["seed"]),),
    )
    _write_csv(output_dir / "seed_metrics.csv", seed_rows)

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(record[field] for field in GROUP_FIELDS)
        groups[key].append(record)
    summary_rows = []
    for key, group in sorted(groups.items()):
        values = np.asarray([row["accuracy"] for row in group], dtype=np.float64)
        ci_low, ci_high = bootstrap_mean_ci(
            values,
            samples=int(config["probe"]["bootstrap_samples"]),
            confidence=float(config["probe"]["confidence_level"]),
            seed=_group_seed(key),
        )
        summary_rows.append(
            {
                **dict(zip(GROUP_FIELDS, key, strict=True)),
                "seed_count": len({int(row["seed"]) for row in group}),
                "mean_accuracy": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    _write_csv(output_dir / "summary.csv", summary_rows)

    delta_rows = _compute_deltas(records)
    _write_csv(output_dir / "deltas.csv", delta_rows)
    interaction_rows = _summarize_deltas(delta_rows, config)
    _write_csv(output_dir / "interaction_summary.csv", interaction_rows)
    _plot_interactions(output_dir / "multimodal_gain_by_condition.png", interaction_rows)
    budget_audit = _audit_budgets(config)
    write_json(output_dir / "budget_and_fairness_audit.json", budget_audit)

    manifest = {
        "schema_version": "tcmi_aggregate_v1",
        "evidence_level": evidence_level,
        "source_file_count": len(result_files),
        "record_count": len(records),
        "summary_row_count": len(summary_rows),
        "delta_row_count": len(delta_rows),
        "artifacts": [
            "seed_metrics.csv",
            "summary.csv",
            "deltas.csv",
            "interaction_summary.csv",
            "multimodal_gain_by_condition.png",
            "budget_and_fairness_audit.json",
        ],
        "matrix_completeness": matrix_completeness(records, config),
        "budget_within_limit": budget_audit["within_gpu_hour_budget"],
        "fairness_checks_passed": budget_audit["fairness_checks_passed"],
    }
    write_json(output_dir / "manifest.json", manifest)
    return output_dir


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1 - alpha)),
    )


def _compute_deltas(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_lookup = {}
    for row in records:
        if (
            row["train_mode"] == "image_only"
            and row["representation_scope"] == "image"
        ):
            key = (
                row["architecture"],
                row["condition"],
                int(row["seed"]),
                row["task"],
                row["metric"],
            )
            baseline_lookup[key] = row

    deltas = []
    for row in records:
        if not row["train_mode"].startswith("multimodal_"):
            continue
        key = (
            row["architecture"],
            row["condition"],
            int(row["seed"]),
            row["task"],
            row["metric"],
        )
        baseline = baseline_lookup.get(key)
        if baseline is None:
            continue
        deltas.append(
            {
                "architecture": row["architecture"],
                "train_mode": row["train_mode"],
                "condition": row["condition"],
                "representation_scope": row["representation_scope"],
                "task": row["task"],
                "metric": row["metric"],
                "seed": int(row["seed"]),
                "baseline_train_mode": "image_only",
                "baseline_representation_scope": "image",
                "multimodal_accuracy": float(row["accuracy"]),
                "image_only_accuracy": float(baseline["accuracy"]),
                "delta": float(row["accuracy"] - baseline["accuracy"]),
            }
        )
    return deltas


def _summarize_deltas(
    delta_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in delta_rows:
        key = (
            row["architecture"],
            row["train_mode"],
            row["condition"],
            row["representation_scope"],
            row["task"],
            row["metric"],
        )
        groups[key].append(float(row["delta"]))
    summaries = []
    for key, values_list in sorted(groups.items()):
        values = np.asarray(values_list, dtype=np.float64)
        ci_low, ci_high = bootstrap_mean_ci(
            values,
            samples=int(config["probe"]["bootstrap_samples"]),
            confidence=float(config["probe"]["confidence_level"]),
            seed=_group_seed(key),
        )
        summaries.append(
            {
                "architecture": key[0],
                "train_mode": key[1],
                "condition": key[2],
                "representation_scope": key[3],
                "task": key[4],
                "metric": key[5],
                "seed_count": len(values),
                "mean_delta": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return summaries


def matrix_completeness(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    observed = {
        (
            row["architecture"],
            row["train_mode"],
            row["condition"],
            int(row["seed"]),
            row["representation_scope"],
            row["task"],
            row["metric"],
        )
        for row in records
    }
    expected = set()
    for architecture in config["matrix"]["architectures"]:
        for train_mode in config["matrix"]["train_modes"]:
            for condition in config["matrix"]["conditions"]:
                if train_mode == "multimodal_conflict" and condition != "conflict":
                    continue
                if train_mode == "multimodal_aligned" and condition == "conflict":
                    continue
                for seed in config["matrix"]["seeds"]:
                    for scope in config["matrix"]["representation_scopes"]:
                        required_metrics = {
                            "clean_accuracy",
                            "unseen_combination_accuracy",
                        }
                        if scope in {"image", "fused"}:
                            required_metrics.add("noisy_image_accuracy")
                        if scope in {"text", "fused"}:
                            required_metrics.update(
                                {
                                    "noisy_text_accuracy",
                                    "conflicting_text_accuracy",
                                }
                            )
                        if scope == "fused":
                            required_metrics.update(
                                {
                                    "missing_image_accuracy",
                                    "missing_text_accuracy",
                                }
                            )
                        if condition == "conflict":
                            required_metrics.add("conflict_flip_accuracy")
                        for task in ("entity", "predicate", "direction", "joint_graph"):
                            for metric in required_metrics:
                                expected.add(
                                    (
                                        architecture,
                                        train_mode,
                                        condition,
                                        int(seed),
                                        scope,
                                        task,
                                        metric,
                                    )
                                )
    missing = sorted(expected - observed)
    return {
        "complete": not missing,
        "expected_cells": len(expected),
        "observed_cells": len(observed & expected),
        "missing_cells": [list(cell) for cell in missing],
    }


def _plot_interactions(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = [
        row
        for row in rows
        if (
            row["train_mode"] == "multimodal_aligned"
            or (
                row["train_mode"] == "multimodal_conflict"
                and row["condition"] == "conflict"
            )
        )
        and row["metric"] == "clean_accuracy"
        and (
            (row["task"] == "joint_graph" and row["representation_scope"] == "fused")
            or (row["task"] != "joint_graph" and row["representation_scope"] == "image")
        )
    ]
    if not selected:
        raise AggregationError("没有 aligned clean delta，无法生成主图")
    conditions = ["redundant", "complementary", "irrelevant", "conflict"]
    tasks = sorted({row["task"] for row in selected})
    figure, axes = plt.subplots(
        len(tasks),
        1,
        figsize=(9, max(3, len(tasks) * 2.5)),
        sharex=True,
    )
    if len(tasks) == 1:
        axes = [axes]
    for axis, task in zip(axes, tasks, strict=True):
        for architecture in sorted({row["architecture"] for row in selected}):
            architecture_rows = {
                row["condition"]: row
                for row in selected
                if row["task"] == task and row["architecture"] == architecture
            }
            values = [
                architecture_rows.get(condition, {}).get("mean_delta", np.nan)
                for condition in conditions
            ]
            axis.plot(conditions, values, marker="o", label=architecture)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel(f"{task}\nΔ accuracy")
        axis.legend()
    axes[-1].set_xlabel("information condition")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_seed(key: tuple[Any, ...]) -> int:
    import hashlib

    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _audit_budgets(config: dict[str, Any]) -> dict[str, Any]:
    run_rows = []
    for manifest_path in sorted(output_root(config).glob("runs/*/run_manifest.json")):
        manifest = read_json(manifest_path)
        if manifest["identity"]["evidence_level"] != config["project"]["evidence_level"]:
            continue
        budget_path = manifest_path.parent / "budget.json"
        parameter_path = manifest_path.parent / "parameter_audit.json"
        if manifest.get("status") != "completed":
            continue
        if not budget_path.exists() or not parameter_path.exists():
            continue
        budget = read_json(budget_path)
        parameters = read_json(parameter_path)
        run_rows.append(
            {
                **manifest["identity"],
                **budget,
                **parameters,
                "run_dir": str(manifest_path.parent),
            }
        )

    total_accounted_seconds = 0.0
    for row in run_rows:
        seconds = row.get("gpu_time_seconds")
        if seconds is None:
            seconds = row.get("wall_clock_seconds", 0.0)
        total_accounted_seconds += float(seconds)
    probe_budget_paths = sorted(
        output_root(config).glob("runs/*/probes/*/probe_budget.json")
    )
    for path in probe_budget_paths:
        budget = read_json(path)
        seconds = budget.get("gpu_time_seconds")
        if seconds is None:
            seconds = budget.get("wall_clock_seconds", 0.0)
        total_accounted_seconds += float(seconds)

    by_cell: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in run_rows:
        key = (row["architecture"], row["condition"], int(row["seed"]))
        by_cell[key][row["train_mode"]] = row

    fairness_failures = []
    checked_pairs = 0
    for key, modes in sorted(by_cell.items()):
        aligned = modes.get("multimodal_aligned") or modes.get("multimodal_conflict")
        matched = modes.get("image_only_matched")
        if aligned is None or matched is None:
            continue
        checked_pairs += 1
        fields = (
            "samples",
            "optimizer_updates",
            "image_encodings",
            "text_sequences",
            "text_tokens",
            "trainable_parameters",
        )
        mismatches = {
            field: {
                "multimodal_aligned": aligned[field],
                "image_only_matched": matched[field],
            }
            for field in fields
            if aligned[field] != matched[field]
        }
        if mismatches:
            fairness_failures.append(
                {
                    "architecture": key[0],
                    "condition": key[1],
                    "seed": key[2],
                    "mismatches": mismatches,
                }
            )

    budget_hours = float(config["project"]["gpu_hour_budget"])
    return {
        "schema_version": "tcmi_budget_fairness_audit_v1",
        "completed_run_count": len(run_rows),
        "probe_budget_count": len(probe_budget_paths),
        "total_accounted_gpu_hours": total_accounted_seconds / 3600,
        "gpu_hour_budget": budget_hours,
        "within_gpu_hour_budget": total_accounted_seconds <= budget_hours * 3600,
        "matched_pair_count": checked_pairs,
        "fairness_failures": fairness_failures,
        "fairness_checks_passed": checked_pairs > 0 and not fairness_failures,
        "accounting_rule": (
            "优先累加 CUDA event GPU time；无 GPU time 时保守使用 wall-clock"
        ),
    }
