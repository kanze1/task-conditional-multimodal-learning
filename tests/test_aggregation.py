from __future__ import annotations

import numpy as np

from tcmi.config import output_root
from tcmi.evaluation.aggregate import (
    AggregationError,
    _audit_budgets,
    _load_probe_records,
    bootstrap_mean_ci,
    matrix_completeness,
)
from tcmi.io import write_json
from tcmi.matrix import matrix_cells


def test_bootstrap_interval_is_deterministic() -> None:
    values = np.asarray([0.1, 0.2, 0.4], dtype=np.float64)
    first = bootstrap_mean_ci(values, samples=100, confidence=0.95, seed=7)
    second = bootstrap_mean_ci(values, samples=100, confidence=0.95, seed=7)
    assert first == second
    assert first[0] <= values.mean() <= first[1]


def test_matrix_excludes_invalid_conflict_cross_product(smoke_config: dict) -> None:
    cells = list(matrix_cells(smoke_config))
    conflict_cells = [
        cell for cell in cells if cell["train_mode"] == "multimodal_conflict"
    ]
    assert conflict_cells
    assert {cell["condition"] for cell in conflict_cells} == {"conflict"}
    assert not any(
        cell["train_mode"] == "multimodal_aligned"
        and cell["condition"] == "conflict"
        for cell in cells
    )


def test_empty_matrix_is_incomplete(smoke_config: dict) -> None:
    completeness = matrix_completeness([], smoke_config)
    assert completeness["complete"] is False
    assert completeness["missing_cells"]


def test_budget_audit_reads_evidence_level_from_run_manifest(
    tiny_temp_config: dict,
) -> None:
    shared_budget = {
        "samples": 16,
        "optimizer_updates": 1,
        "image_encodings": 16,
        "text_sequences": 16,
        "text_tokens": 224,
        "wall_clock_seconds": 0.1,
        "gpu_time_seconds": 0.05,
    }
    shared_parameters = {
        "total_parameters": 100,
        "trainable_parameters": 100,
    }
    for train_mode in ("image_only_matched", "multimodal_aligned"):
        run_dir = output_root(tiny_temp_config) / "runs" / train_mode
        run_dir.mkdir(parents=True)
        write_json(
            run_dir / "run_manifest.json",
            {
                "status": "completed",
                "evidence_level": "smoke",
                "identity": {
                    "architecture": "tiny_cnn",
                    "train_mode": train_mode,
                    "condition": "redundant",
                    "seed": 2601,
                },
            },
        )
        write_json(run_dir / "budget.json", shared_budget)
        write_json(run_dir / "parameter_audit.json", shared_parameters)

    audit = _audit_budgets(tiny_temp_config)
    assert audit["completed_run_count"] == 2
    assert audit["matched_pair_count"] == 1
    assert audit["fairness_checks_passed"] is True
    assert audit["fairness_failures"] == []


def test_load_probe_records_uses_completed_manifest_evidence_level(
    tiny_temp_config: dict,
) -> None:
    probe_dir = (
        output_root(tiny_temp_config)
        / "runs"
        / "sample-run"
        / "probes"
        / "image"
    )
    metrics_path = probe_dir / "probe_metrics.json"
    write_json(
        probe_dir / "probe_run_manifest.json",
        {
            "status": "completed",
            "evidence_level": "smoke",
            "metrics_path": metrics_path.name,
        },
    )
    write_json(
        metrics_path,
        {
            "schema_version": "tcmi_probe_metrics_v1",
            "metrics": [
                {
                    "architecture": "tiny_cnn",
                    "train_mode": "image_only",
                    "condition": "redundant",
                    "seed": 2601,
                    "representation_scope": "image",
                    "task": "entity",
                    "metric": "clean_accuracy",
                    "accuracy": 0.5,
                }
            ],
        },
    )

    records = _load_probe_records(tiny_temp_config, [metrics_path])

    assert records[0]["evidence_level"] == "smoke"
    assert records[0]["source_path"] == str(metrics_path)


def test_load_probe_records_rejects_payload_manifest_evidence_mismatch(
    tiny_temp_config: dict,
) -> None:
    probe_dir = (
        output_root(tiny_temp_config)
        / "runs"
        / "sample-run"
        / "probes"
        / "image"
    )
    metrics_path = probe_dir / "probe_metrics.json"
    write_json(
        probe_dir / "probe_run_manifest.json",
        {
            "status": "completed",
            "evidence_level": "smoke",
            "metrics_path": metrics_path.name,
        },
    )
    write_json(
        metrics_path,
        {
            "schema_version": "tcmi_probe_metrics_v1",
            "evidence_level": "formal",
            "metrics": [],
        },
    )

    with np.testing.assert_raises(AggregationError):
        _load_probe_records(tiny_temp_config, [metrics_path])
