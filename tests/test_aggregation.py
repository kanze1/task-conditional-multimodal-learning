from __future__ import annotations

import numpy as np

from tcmi.evaluation.aggregate import bootstrap_mean_ci, matrix_completeness
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
