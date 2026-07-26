from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tcmi.config import output_root
from tcmi.io import read_json, write_json


class DecisionError(RuntimeError):
    """Raised when a Go/No-Go decision cannot be audited."""


def render_decision(config: dict[str, Any]) -> Path:
    evidence_level = config["project"]["evidence_level"]
    aggregate_dir = output_root(config) / "aggregates" / evidence_level
    manifest = read_json(aggregate_dir / "manifest.json")
    rows = _read_csv(aggregate_dir / "interaction_summary.csv")

    if evidence_level != "formal" or config["project"]["protocol_status"] != "frozen":
        decision = {
            "schema_version": "tcmi_decision_v1",
            "decision": "experimental_in_progress",
            "reason": "只有 frozen protocol 下的 formal 完整矩阵可以触发 Go/No-Go",
            "evidence_level": evidence_level,
            "protocol_status": config["project"]["protocol_status"],
            "matrix_completeness": manifest["matrix_completeness"],
            "go_no_go_evaluated": False,
        }
    elif not manifest["matrix_completeness"]["complete"]:
        decision = {
            "schema_version": "tcmi_decision_v1",
            "decision": "experimental_in_progress",
            "reason": "formal 主矩阵不完整",
            "evidence_level": evidence_level,
            "protocol_status": config["project"]["protocol_status"],
            "matrix_completeness": manifest["matrix_completeness"],
            "go_no_go_evaluated": False,
        }
    elif not manifest.get("budget_within_limit", False):
        decision = {
            "schema_version": "tcmi_decision_v1",
            "decision": "no_go",
            "reason": "formal 阶段累计 GPU 时间超过预注册预算",
            "evidence_level": evidence_level,
            "protocol_status": config["project"]["protocol_status"],
            "matrix_completeness": manifest["matrix_completeness"],
            "go_no_go_evaluated": True,
        }
    elif not manifest.get("fairness_checks_passed", False):
        decision = {
            "schema_version": "tcmi_decision_v1",
            "decision": "experimental_in_progress",
            "reason": "公平预算核对未通过或缺失",
            "evidence_level": evidence_level,
            "protocol_status": config["project"]["protocol_status"],
            "matrix_completeness": manifest["matrix_completeness"],
            "go_no_go_evaluated": False,
        }
    else:
        decision = _evaluate_formal_thresholds(rows, manifest)

    output_path = aggregate_dir / "decision.json"
    write_json(output_path, decision)
    return output_path


def _evaluate_formal_thresholds(
    rows: list[dict[str, str]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    aligned_clean = [
        row
        for row in rows
        if row["train_mode"] == "multimodal_aligned"
        and row["metric"] == "clean_accuracy"
    ]
    complementary = [
        row
        for row in aligned_clean
        if row["condition"] == "complementary"
        and (
            (row["task"] == "joint_graph" and row["representation_scope"] == "fused")
            or (row["task"] != "joint_graph" and row["representation_scope"] == "image")
        )
    ]
    task_passes: dict[str, bool] = {}
    for row in complementary:
        task_passes.setdefault(row["task"], True)
        mean_delta = float(row["mean_delta"])
        sample_std = float(row["sample_std"])
        task_passes[row["task"]] &= mean_delta >= 0.05 and mean_delta > 2 * sample_std

    architecture_directions: dict[str, set[int]] = {}
    for row in complementary:
        sign = 1 if float(row["mean_delta"]) > 0 else -1
        architecture_directions.setdefault(row["task"], set()).add(sign)
    direction_consistent = all(
        len(signs) == 1 and 1 in signs for signs in architecture_directions.values()
    )
    passed_task_count = sum(task_passes.values())
    checks = {
        "complementary_two_tasks": passed_task_count >= 2,
        "architecture_direction_consistency": direction_consistent,
    }
    # irrelevant、shuffled 与 conflict-flip 的正式统计定义仍由冻结协议提供；
    # 若没有机器可读阈值，不允许自动宣告 Go。
    checks["all_machine_readable_thresholds_available"] = False
    return {
        "schema_version": "tcmi_decision_v1",
        "decision": "experimental_in_progress",
        "reason": "部分正式阈值尚未机器可读化，禁止自动宣告 Go",
        "go_no_go_evaluated": False,
        "checks": checks,
        "task_passes": task_passes,
        "matrix_completeness": manifest["matrix_completeness"],
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
