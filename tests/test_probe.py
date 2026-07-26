from __future__ import annotations

import torch
from torch import nn

from tcmi.evaluation.features import FeatureSet
from tcmi.evaluation.probe import _evaluate_classifier, _failure_examples


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
