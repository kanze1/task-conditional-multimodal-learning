from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import tcmi.evaluation.probe as probe_module
from tcmi.config import public_config, stable_hash
from tcmi.evaluation.features import FeatureSet
from tcmi.evaluation.probe import (
    _evaluate_classifier,
    _failure_examples,
    _probe_provenance,
)
from tcmi.io import sha256_file, write_json


def test_probe_evaluation_respects_unseen_mask() -> None:
    classifier = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        classifier.weight.copy_(torch.eye(2))
    feature_set = FeatureSet(
        features=torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]]),
        labels=torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
        unseen_mask=torch.tensor([False, True, True]),
        sample_ids=["test-0", "test-1", "test-2"],
    )
    full = _evaluate_classifier(
        classifier,
        feature_set,
        task_index=0,
        device=torch.device("cpu"),
    )
    unseen = _evaluate_classifier(
        classifier,
        feature_set,
        task_index=0,
        device=torch.device("cpu"),
        mask=feature_set.unseen_mask,
    )
    assert full["accuracy"] == 2 / 3
    assert unseen["accuracy"] == 1 / 2


def test_failure_examples_keep_sample_provenance() -> None:
    classifier = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        classifier.weight.copy_(torch.eye(2))
    feature_set = FeatureSet(
        features=torch.tensor([[3.0, 0.0]]),
        labels=torch.tensor([[1, 0, 0, 0]]),
        unseen_mask=torch.tensor([True]),
        sample_ids=["test-00000001"],
    )
    failures = _failure_examples(
        classifier=classifier,
        feature_set=feature_set,
        task_index=0,
        task="entity",
        device=torch.device("cpu"),
        limit=5,
        identity={
            "architecture": "tiny_cnn",
            "train_mode": "image_only",
            "condition": "redundant",
            "seed": 2601,
            "evidence_level": "smoke",
        },
        representation_scope="image",
    )
    assert failures[0]["sample_id"] == "test-00000001"
    assert failures[0]["is_unseen_combination"] is True


def test_probe_provenance_records_evaluation_and_training_sources(
    tiny_temp_config: dict,
    monkeypatch,
) -> None:
    project_root = Path(tiny_temp_config["_meta"]["config_path"]).parent.parent
    protocol_path = project_root / tiny_temp_config["project"]["protocol_path"]
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("frozen protocol\n", encoding="utf-8")

    run_dir = project_root / "artifacts" / "runs" / "sample-run"
    checkpoint_path = run_dir / "checkpoints" / "epoch_0001.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    config_hash = stable_hash(public_config(tiny_temp_config))
    write_json(
        run_dir / "run_manifest.json",
        {
            "status": "completed",
            "evidence_level": "smoke",
            "identity": {
                "architecture": "tiny_cnn",
                "train_mode": "image_only",
                "condition": "redundant",
                "seed": 2601,
            },
            "config_hash": config_hash,
            "dataset_manifest_hash": "dataset-hash",
            "final_checkpoint": "checkpoints/epoch_0001.pt",
            "source": {"git_commit": "training-commit"},
        },
    )
    monkeypatch.setattr(
        probe_module,
        "dataset_manifest_hash",
        lambda config: "dataset-hash",
    )
    monkeypatch.setattr(
        probe_module,
        "source_snapshot",
        lambda root: {
            "git_commit": "evaluation-commit",
            "dirty": False,
            "dirty_paths": [],
        },
    )

    provenance = _probe_provenance(tiny_temp_config, run_dir, "image")
    assert provenance["config_hash"] == config_hash
    assert provenance["dataset_manifest_hash"] == "dataset-hash"
    assert provenance["protocol_sha256"] == sha256_file(protocol_path)
    assert provenance["source"]["git_commit"] == "evaluation-commit"
    assert provenance["training_run"]["git_commit"] == "training-commit"
    assert provenance["checkpoint"]["checkpoint_sha256"] == sha256_file(
        checkpoint_path
    )
