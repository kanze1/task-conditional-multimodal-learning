from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from tcmi.constants import INFORMATION_CONDITIONS, TASKS
from tcmi.data.generator import dataset_root
from tcmi.data.schema import SceneGraph
from tcmi.io import read_json


def run_oracle_audit(config: dict[str, Any]) -> dict[str, Any]:
    root = dataset_root(config)
    results = []
    failures: list[str] = []
    for condition in INFORMATION_CONDITIONS:
        manifest = read_json(root / condition / "manifest.json")
        train_rows = _load_rows(root, manifest, "train")
        test_rows = _load_rows(root, manifest, "test")
        for modality in ("image", "text", "joint"):
            for task_index, task in enumerate(TASKS):
                accuracy = _lookup_oracle_accuracy(
                    train_rows=train_rows,
                    test_rows=test_rows,
                    condition=condition,
                    modality=modality,
                    task=task,
                    task_index=task_index,
                )
                results.append(
                    {
                        "condition": condition,
                        "modality": modality,
                        "task": task,
                        "test_accuracy": accuracy,
                        "identifiability_accuracy": _collision_oracle_accuracy(
                            rows=train_rows + test_rows,
                            condition=condition,
                            modality=modality,
                            task=task,
                            task_index=task_index,
                        ),
                        "test_sample_count": len(test_rows),
                    }
                )

    lookup = {
        (row["condition"], row["modality"], row["task"]): row
        for row in results
    }
    for modality in ("image", "text", "joint"):
        for task in TASKS:
            if (
                lookup[("redundant", modality, task)]["identifiability_accuracy"]
                < 0.99
            ):
                failures.append(f"redundant oracle 未接近完美: {modality}/{task}")
    if (
        lookup[("complementary", "joint", "joint_graph")][
            "identifiability_accuracy"
        ]
        < 0.99
    ):
        failures.append("complementary 的 joint_graph 联合 oracle 不可辨识")
    complementary_image = lookup[("complementary", "image", "joint_graph")]
    if complementary_image["identifiability_accuracy"] > _chance_ceiling(
        "joint_graph", complementary_image["test_sample_count"]
    ):
        failures.append("complementary 的 image 单模态泄漏 text_key")
    complementary_text = lookup[("complementary", "text", "joint_graph")]
    if complementary_text["identifiability_accuracy"] > _chance_ceiling(
        "joint_graph", complementary_text["test_sample_count"]
    ):
        failures.append("complementary 的 text 单模态泄漏 visual_key")
    for task in TASKS:
        row = lookup[("irrelevant", "text", task)]
        if row["identifiability_accuracy"] > _chance_ceiling(
            task, row["test_sample_count"]
        ):
            failures.append(f"irrelevant 文本异常预测 {task}")
    for task in ("predicate", "direction", "joint_graph"):
        row = lookup[("conflict", "text", task)]
        if row["test_accuracy"] > _chance_ceiling(task, row["test_sample_count"]):
            failures.append(f"conflict test 文本未按预期翻转 {task}")

    return {
        "schema_version": "tcmi_oracle_audit_v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "results": results,
        "method": (
            "按预注册投影可见槽构造离散特征，在 train 上拟合 majority lookup，"
            "在 test 上验证有限样本可辨识性"
        ),
    }


def _load_rows(
    root: Any,
    manifest: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for shard in manifest["splits"][split]["shards"]:
        with np.load(root / shard["path"], allow_pickle=False) as arrays:
            for index in range(len(arrays["sample_ids"])):
                graph = SceneGraph(
                    *[int(value) for value in arrays["latent_fields"][index]]
                )
                rows.append(
                    {
                        "graph": graph,
                        "input_ids": arrays["input_ids"][index].astype(int).tolist(),
                        "labels": arrays["labels"][index].astype(int).tolist(),
                    }
                )
    return rows


def _lookup_oracle_accuracy(
    *,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    condition: str,
    modality: str,
    task: str,
    task_index: int,
) -> float:
    counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    global_counts: Counter[int] = Counter()
    for row in train_rows:
        label = int(row["labels"][task_index])
        feature = _oracle_feature(row, condition, modality, task)
        counts[feature][label] += 1
        global_counts[label] += 1
    default_label = global_counts.most_common(1)[0][0]
    correct = 0
    for row in test_rows:
        feature = _oracle_feature(row, condition, modality, task)
        prediction = (
            counts[feature].most_common(1)[0][0] if feature in counts else default_label
        )
        correct += prediction == int(row["labels"][task_index])
    return correct / len(test_rows)


def _collision_oracle_accuracy(
    *,
    rows: list[dict[str, Any]],
    condition: str,
    modality: str,
    task: str,
    task_index: int,
) -> float:
    counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    for row in rows:
        feature = _oracle_feature(row, condition, modality, task)
        counts[feature][int(row["labels"][task_index])] += 1
    correct = sum(counter.most_common(1)[0][1] for counter in counts.values())
    return correct / len(rows)


def _oracle_feature(
    row: dict[str, Any],
    condition: str,
    modality: str,
    task: str,
) -> tuple[int, ...]:
    graph: SceneGraph = row["graph"]
    image_features = {
        "entity": (graph.subject_shape, graph.object_shape),
        "predicate": (graph.predicate,),
        "direction": (graph.direction,),
        "joint_graph": (
            (graph.visual_key,)
            if condition == "complementary"
            else (graph.visual_key, graph.text_key)
        ),
    }[task]
    value_positions = {
        "entity": (2, 4),
        "predicate": (6,),
        "direction": (8,),
        "joint_graph": (10, 12),
    }[task]
    if condition == "complementary":
        value_positions = (12,) if task == "joint_graph" else ()
    elif condition == "irrelevant":
        value_positions = ()
    text_features = tuple(int(row["input_ids"][position]) for position in value_positions)

    if modality == "image":
        return image_features
    if modality == "text":
        return text_features
    if modality == "joint":
        return image_features + text_features
    raise ValueError(f"未知 oracle modality: {modality}")


def _chance_ceiling(task: str, sample_count: int) -> float:
    class_count = {
        "entity": 16,
        "predicate": 4,
        "direction": 2,
        "joint_graph": 2,
    }[task]
    chance = 1 / class_count
    standard_error = (chance * (1 - chance) / max(1, sample_count)) ** 0.5
    return min(1.0, chance + 3 * standard_error)
