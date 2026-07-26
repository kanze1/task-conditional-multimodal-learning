from __future__ import annotations

from collections import Counter
from typing import Any

from tcmi.config import output_root
from tcmi.io import read_json


def collect_status(config: dict[str, Any]) -> dict[str, Any]:
    root = output_root(config)
    data_manifests = sorted(root.glob("data/*/manifest.json"))
    audit_reports = sorted(root.glob("data/*/audit_report.json"))
    run_manifests = sorted(root.glob("runs/*/run_manifest.json"))
    run_statuses = Counter()
    completed_probe_count = 0
    for path in run_manifests:
        manifest = read_json(path)
        run_statuses[manifest.get("status", "unknown")] += 1
        completed_probe_count += len(manifest.get("completed_probes", []))
    probe_statuses = Counter()
    for path in sorted(root.glob("runs/*/probes/*/probe_run_manifest.json")):
        probe_statuses[read_json(path).get("status", "unknown")] += 1
    aggregate_manifests = sorted(root.glob("aggregates/*/manifest.json"))
    decisions = sorted(root.glob("aggregates/*/decision.json"))
    return {
        "output_root": str(root),
        "dataset_manifest_count": len(data_manifests),
        "passed_audit_count": sum(
            read_json(path).get("status") == "passed" for path in audit_reports
        ),
        "run_count": len(run_manifests),
        "run_statuses": dict(sorted(run_statuses.items())),
        "completed_probe_count": completed_probe_count,
        "probe_statuses": dict(sorted(probe_statuses.items())),
        "aggregate_count": len(aggregate_manifests),
        "decision_count": len(decisions),
        "paths": {
            "datasets": [str(path) for path in data_manifests],
            "aggregates": [str(path) for path in aggregate_manifests],
            "decisions": [str(path) for path in decisions],
        },
    }
