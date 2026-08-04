from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from tcmi.constants import (
    COMPLEMENTARY_PRIMARY_TASKS,
    INFORMATION_CONDITIONS,
    TASK_CLASS_COUNTS,
    TASKS,
)
from tcmi.data.generator import complementary_object_shape, dataset_root
from tcmi.data.schema import SceneGraph
from tcmi.io import read_json


def run_oracle_audit(config: dict[str, Any]) -> dict[str, Any]:
    root = dataset_root(config)
    key_correlation = float(config["data"].get("key_correlation", 0.5))
    results = []
    failures: list[str] = []
    condition_rows: dict[str, list[dict[str, Any]]] = {}
    for condition in INFORMATION_CONDITIONS:
        manifest = read_json(root / condition / "manifest.json")
        train_rows = _load_rows(root, manifest, "train")
        test_rows = _load_rows(root, manifest, "test")
        condition_rows[condition] = train_rows + test_rows
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
    linear_decodability = []
    for task in COMPLEMENTARY_PRIMARY_TASKS:
        joint = lookup[("complementary", "joint", task)]
        if joint["identifiability_accuracy"] < 0.99:
            failures.append(f"complementary 的 {task} 联合 oracle 不可辨识")
        for modality in ("image", "text"):
            row = lookup[("complementary", modality, task)]
            if row["identifiability_accuracy"] > _partial_information_ceiling(
                task,
                row["test_sample_count"],
                key_correlation,
            ):
                failures.append(
                    f"complementary 的 {modality} 单模态泄漏完整 {task} 标签"
                )
        linear_accuracy = _fixed_linear_pair_accuracy(
            condition_rows["complementary"],
            task,
        )
        linear_decodability.append(
            {
                "condition": "complementary",
                "task": task,
                "representation_scope": "fused",
                "accuracy": linear_accuracy,
                "feature_geometry": "concatenated_one_hot_pair",
                "class_count": TASK_CLASS_COUNTS[task],
            }
        )
        if linear_accuracy < 1.0:
            failures.append(f"complementary 的 {task} 不满足线性联合可解码契约")
    for task in ("predicate", "direction"):
        for modality in ("image", "text"):
            if (
                lookup[("complementary", modality, task)][
                    "identifiability_accuracy"
                ]
                < 0.99
            ):
                failures.append(
                    f"complementary 的 {modality} 缺少共享对齐锚点 {task}"
                )
    for task in TASKS:
        row = lookup[("irrelevant", "text", task)]
        if row["identifiability_accuracy"] > _chance_ceiling(
            task, row["test_sample_count"], key_correlation
        ):
            failures.append(f"irrelevant 文本异常预测 {task}")
    for task in ("predicate", "direction", "joint_graph"):
        row = lookup[("conflict", "text", task)]
        if row["test_accuracy"] > _chance_ceiling(
            task, row["test_sample_count"], key_correlation
        ):
            failures.append(f"conflict test 文本未按预期翻转 {task}")

    return {
        "schema_version": "tcmi_oracle_audit_v2",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "results": results,
        "linear_decodability": linear_decodability,
        "method": (
            "按预注册投影可见槽构造离散特征，在 train 上拟合 majority lookup，"
            "在 test 上验证有限样本可辨识性；对互补主任务额外使用固定的"
            "拼接 one-hot 多类线性分类器验证表示几何"
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
        "entity": (
            (graph.subject_shape, complementary_object_shape(graph))
            if condition == "complementary"
            else (graph.subject_shape, graph.object_shape)
        ),
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
        value_positions = {
            "entity": (4,),
            "predicate": (6,),
            "direction": (8,),
            "joint_graph": (12,),
        }[task]
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


def _chance_ceiling(
    task: str,
    sample_count: int,
    key_correlation: float = 0.5,
) -> float:
    # key_correlation != 0.5 时 joint_graph 边缘分布不再均匀，最大类概率为 rho/2。
    if task == "joint_graph":
        chance = key_correlation / 2
    else:
        chance = 1 / TASK_CLASS_COUNTS[task]
    standard_error = (chance * (1 - chance) / max(1, sample_count)) ** 0.5
    return min(1.0, chance + 3 * standard_error)


def _partial_information_ceiling(
    task: str,
    sample_count: int,
    key_correlation: float = 0.5,
) -> float:
    # 单模态知道一个 key 时，另一 key 的最优预测准确率等于 key_correlation。
    partial_accuracy = {
        "entity": 0.25,
        "joint_graph": key_correlation,
    }[task]
    standard_error = (
        partial_accuracy * (1 - partial_accuracy) / max(1, sample_count)
    ) ** 0.5
    return min(1.0, partial_accuracy + 3 * standard_error)


def _fixed_linear_pair_accuracy(rows: list[dict[str, Any]], task: str) -> float:
    if task == "entity":
        cardinalities = (4, 4)
    elif task == "joint_graph":
        cardinalities = (2, 2)
    else:
        raise ValueError(f"未知互补主任务: {task}")

    expected_class_count = cardinalities[0] * cardinalities[1]
    if TASK_CLASS_COUNTS[task] != expected_class_count:
        return 0.0

    task_index = TASKS.index(task)
    correct = 0
    for row in rows:
        graph = row["graph"]
        if task == "entity":
            left_value, right_value = graph.subject_shape, graph.object_shape
        else:
            left_value, right_value = graph.visual_key, graph.text_key
        scores = [
            int(class_index // cardinalities[1] == left_value)
            + int(class_index % cardinalities[1] == right_value)
            for class_index in range(expected_class_count)
        ]
        prediction = int(np.argmax(np.asarray(scores)))
        correct += prediction == int(row["labels"][task_index])
    return correct / len(rows)
