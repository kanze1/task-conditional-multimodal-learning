from __future__ import annotations

from tcmi.data.audit import audit_dataset
from tcmi.data.generator import generate_dataset
from tcmi.io import read_json


def test_generated_dataset_passes_mechanical_audit(tiny_temp_config: dict) -> None:
    generate_dataset(tiny_temp_config)
    report_path = audit_dataset(tiny_temp_config)
    report = read_json(report_path)
    assert report["schema_version"] == "tcmi_data_audit_v2"
    assert report["oracle_audit"]["schema_version"] == "tcmi_oracle_audit_v2"
    assert report["status"] == "passed"
    assert report["failures"] == []
    assert set(report["condition_summaries"]) == {
        "redundant",
        "complementary",
        "irrelevant",
        "conflict",
    }
    assert report["checks"]["linear_probe_identifiability"] is True
    assert report["checks"]["complementary_alignment_anchor"] is True
    linear_results = {
        row["task"]: row for row in report["oracle_audit"]["linear_decodability"]
    }
    assert set(linear_results) == {"entity", "joint_graph"}
    assert all(row["accuracy"] == 1.0 for row in linear_results.values())
