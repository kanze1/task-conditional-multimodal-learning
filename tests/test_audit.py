from __future__ import annotations

from tcmi.data.audit import audit_dataset
from tcmi.data.generator import generate_dataset
from tcmi.io import read_json


def test_generated_dataset_passes_mechanical_audit(tiny_temp_config: dict) -> None:
    generate_dataset(tiny_temp_config)
    report_path = audit_dataset(tiny_temp_config)
    report = read_json(report_path)
    assert report["status"] == "passed"
    assert report["failures"] == []
    assert set(report["condition_summaries"]) == {
        "redundant",
        "complementary",
        "irrelevant",
        "conflict",
    }
