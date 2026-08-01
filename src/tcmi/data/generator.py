from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tcmi.config import canonical_json, output_root, stable_hash
from tcmi.constants import INFORMATION_CONDITIONS, SPLITS
from tcmi.data.schema import LABEL_COLUMNS, SceneGraph, labels_as_tuple
from tcmi.data.vocab import SLOT_NAMES, VOCABULARY
from tcmi.io import sha256_file, write_json, write_jsonl

SHAPE_COLORS = np.asarray(
    [
        [220, 70, 70],
        [70, 180, 90],
        [70, 120, 230],
        [225, 180, 55],
    ],
    dtype=np.uint8,
)

PREDICATE_COLORS = np.asarray(
    [
        [240, 240, 240],
        [230, 100, 220],
        [80, 220, 220],
        [240, 140, 60],
    ],
    dtype=np.uint8,
)


class GenerationError(RuntimeError):
    """Raised when deterministic data generation cannot satisfy its contract."""


def dataset_root(config: dict[str, Any]) -> Path:
    data_hash = stable_hash(config["data"])[:12]
    dataset_id = (
        f"{config['data']['dataset_name']}__{config['data']['schema_version']}__{data_hash}"
    )
    return output_root(config) / "data" / dataset_id


def sample_graph(
    generation_seed: int,
    split: str,
    index: int,
    heldout_modulus: int,
) -> SceneGraph:
    desired_unseen = split != "train" and index % 2 == 1
    for attempt in range(10_000):
        rng = _rng_for(generation_seed, split, index, attempt)
        graph = SceneGraph(
            subject_shape=int(rng.integers(0, 4)),
            object_shape=int(rng.integers(0, 4)),
            predicate=int(rng.integers(0, 4)),
            direction=int(rng.integers(0, 2)),
            visual_key=int(rng.integers(0, 2)),
            text_key=int(rng.integers(0, 2)),
            layout_jitter_x=int(rng.integers(-3, 4)),
            layout_jitter_y=int(rng.integers(-3, 4)),
            template_id=int(rng.integers(0, 8)),
            scene_nonce=int(rng.integers(0, 2**31 - 1)),
        )
        heldout = is_unseen_combination(graph, heldout_modulus)
        if split == "train" and not heldout:
            return graph
        if split != "train" and heldout == desired_unseen:
            return graph
    raise GenerationError(f"无法为 {split}:{index} 采样满足组合约束的图")


def is_unseen_combination(graph: SceneGraph, heldout_modulus: int) -> bool:
    return (graph.entity_label * 3 + graph.predicate) % heldout_modulus == 0


def complementary_object_shape(graph: SceneGraph) -> int:
    """Return an object-shape nuisance independent of the true object label."""
    return graph.scene_nonce % 4


def _rng_for(seed: int, split: str, index: int, attempt: int = 0) -> np.random.Generator:
    split_id = SPLITS.index(split)
    seed_sequence = np.random.SeedSequence([seed, split_id, index, attempt])
    return np.random.default_rng(seed_sequence)


def render_image(graph: SceneGraph, condition: str, image_size: int) -> np.ndarray:
    if condition not in INFORMATION_CONDITIONS:
        raise GenerationError(f"未知信息条件: {condition}")
    canvas = np.full((3, image_size, image_size), 24, dtype=np.uint8)
    center_y = image_size // 2 + graph.layout_jitter_y
    left_x = image_size // 4 + graph.layout_jitter_x
    right_x = image_size * 3 // 4 + graph.layout_jitter_x
    subject_x, object_x = (
        (left_x, right_x) if graph.direction == 0 else (right_x, left_x)
    )

    rendered_object_shape = (
        complementary_object_shape(graph)
        if condition == "complementary"
        else graph.object_shape
    )
    _draw_relation(canvas, subject_x, object_x, center_y, graph.predicate)
    _draw_shape(canvas, subject_x, center_y, graph.subject_shape)
    _draw_shape(canvas, object_x, center_y, rendered_object_shape)
    _draw_key_patch(canvas, 2, image_size - 5, graph.visual_key)

    # complementary 中 text_key 只存在于文本；其余条件让 image-only 保持完整可辨。
    if condition != "complementary":
        _draw_key_patch(canvas, image_size - 5, image_size - 5, graph.text_key)
    return canvas


def _draw_shape(canvas: np.ndarray, center_x: int, center_y: int, shape: int) -> None:
    color = SHAPE_COLORS[shape]
    radius = 4
    height, width = canvas.shape[1:]
    for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
        for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
            dx = x - center_x
            dy = y - center_y
            inside = (
                dx * dx + dy * dy <= radius * radius
                if shape == 0
                else abs(dx) <= radius and abs(dy) <= radius
                if shape == 1
                else abs(dx) + abs(dy) <= radius
                if shape == 2
                else dy >= -radius and abs(dx) <= radius - max(0, dy)
            )
            if inside:
                canvas[:, y, x] = color


def _draw_relation(
    canvas: np.ndarray,
    subject_x: int,
    object_x: int,
    center_y: int,
    predicate: int,
) -> None:
    color = PREDICATE_COLORS[predicate]
    start, end = sorted((subject_x, object_x))
    for x in range(start + 4, end - 3):
        if predicate == 1 and x % 2 == 0:
            continue
        thickness = 1 + int(predicate == 3)
        for offset in range(-thickness + 1, thickness):
            canvas[:, center_y + offset, x] = color
    arrow_sign = 1 if object_x > subject_x else -1
    tip_x = object_x - arrow_sign * 5
    for offset in range(3):
        x = tip_x - arrow_sign * offset
        canvas[:, center_y - offset, x] = color
        canvas[:, center_y + offset, x] = color


def _draw_key_patch(canvas: np.ndarray, center_x: int, center_y: int, value: int) -> None:
    color = np.asarray([245, 245, 245] if value else [65, 65, 65], dtype=np.uint8)
    x_start = max(0, center_x - 2)
    x_end = min(canvas.shape[2], center_x + 2)
    y_start = max(0, center_y - 2)
    y_end = min(canvas.shape[1], center_y + 2)
    canvas[:, y_start:y_end, x_start:x_end] = color[:, None, None]


def text_token_ids(
    graph: SceneGraph,
    condition: str,
    split: str,
    generation_seed: int,
    index: int,
) -> list[int]:
    true_values = {
        "subject": f"shape_{graph.subject_shape}",
        "object": f"shape_{graph.object_shape}",
        "predicate": f"predicate_{graph.predicate}",
        "direction": f"direction_{graph.direction}",
        "visual_key": f"visual_key_{graph.visual_key}",
        "text_key": f"text_key_{graph.text_key}",
    }
    rng = _rng_for(generation_seed + 97, split, index)
    nuisance_values = {
        slot: f"nuisance_{int(rng.integers(0, 64))}" for slot in SLOT_NAMES
    }

    if condition == "redundant":
        values = true_values
    elif condition == "complementary":
        values = {
            **nuisance_values,
            "object": true_values["object"],
            "predicate": true_values["predicate"],
            "direction": true_values["direction"],
            "text_key": true_values["text_key"],
        }
    elif condition == "irrelevant":
        values = nuisance_values
    elif condition == "conflict":
        if split in {"train", "validation"}:
            values = true_values
        else:
            values = {
                "subject": f"shape_{(graph.subject_shape + 1) % 4}",
                "object": f"shape_{(graph.object_shape + 2) % 4}",
                "predicate": f"predicate_{(graph.predicate + 1) % 4}",
                "direction": f"direction_{1 - graph.direction}",
                "visual_key": f"visual_key_{1 - graph.visual_key}",
                "text_key": f"text_key_{graph.text_key}",
            }
    else:
        raise GenerationError(f"未知信息条件: {condition}")

    token_ids = [VOCABULARY.bos_id]
    for slot in SLOT_NAMES:
        token_ids.append(VOCABULARY.token_id(slot))
        token_ids.append(VOCABULARY.token_id(values[slot]))
    token_ids.append(VOCABULARY.eos_id)
    return token_ids


def generate_dataset(config: dict[str, Any]) -> Path:
    root = dataset_root(config)
    if root.exists():
        raise GenerationError(f"数据目录已存在，拒绝覆盖: {root}")
    root.mkdir(parents=True)

    data_config = config["data"]
    generation_seed = int(data_config["generation_seed"])
    heldout_modulus = int(data_config["heldout_modulus"])
    shard_size = int(data_config["shard_size"])
    image_size = int(data_config["image_size"])
    preview_count = int(data_config["preview_samples_per_split"])

    condition_entries: dict[str, dict[str, str]] = {}
    reference_latent_digests: dict[str, str] | None = None

    for condition in data_config["conditions"]:
        condition_dir = root / condition
        condition_dir.mkdir()
        split_entries: dict[str, Any] = {}
        latent_digests: dict[str, str] = {}
        previews: list[dict[str, Any]] = []

        for split, split_size_raw in data_config["splits"].items():
            split_size = int(split_size_raw)
            split_dir = condition_dir / split
            split_dir.mkdir()
            latent_digest = hashlib.sha256()
            shard_entries = []

            for shard_index, batch in enumerate(
                _graph_batches(
                    generation_seed,
                    split,
                    split_size,
                    shard_size,
                    heldout_modulus,
                )
            ):
                start_index, graphs = batch
                arrays = _build_shard(
                    graphs=graphs,
                    start_index=start_index,
                    split=split,
                    condition=condition,
                    generation_seed=generation_seed,
                    heldout_modulus=heldout_modulus,
                    image_size=image_size,
                )
                for graph in graphs:
                    latent_digest.update(
                        (canonical_json(graph.to_dict()) + "\n").encode("utf-8")
                    )
                shard_name = f"shard_{shard_index:05d}.npz"
                shard_path = split_dir / shard_name
                np.savez_compressed(shard_path, **arrays)
                shard_entries.append(
                    {
                        "path": str(shard_path.relative_to(root)).replace("\\", "/"),
                        "count": len(graphs),
                        "sha256": sha256_file(shard_path),
                    }
                )

                if len(previews) < preview_count * len(data_config["splits"]):
                    for local_index, graph in enumerate(graphs):
                        if len(
                            [preview for preview in previews if preview["split"] == split]
                        ) >= preview_count:
                            break
                        token_ids = arrays["input_ids"][local_index].tolist()
                        previews.append(
                            {
                                "sample_id": str(arrays["sample_ids"][local_index]),
                                "split": split,
                                "condition": condition,
                                "graph": graph.to_dict(),
                                "labels": graph.labels,
                                "caption_tokens": VOCABULARY.decode(token_ids),
                                "is_unseen_combination": bool(
                                    arrays["is_unseen_combination"][local_index]
                                ),
                            }
                        )

            latent_digests[split] = latent_digest.hexdigest()
            split_entries[split] = {
                "count": split_size,
                "shards": shard_entries,
            }

        if reference_latent_digests is None:
            reference_latent_digests = latent_digests
        elif latent_digests != reference_latent_digests:
            raise GenerationError("不同信息条件生成了不一致的潜在图")

        condition_manifest = {
            "schema_version": data_config["schema_version"],
            "condition": condition,
            "generation_seed": generation_seed,
            "label_columns": list(LABEL_COLUMNS),
            "latent_digests": latent_digests,
            "splits": split_entries,
        }
        condition_manifest_path = condition_dir / "manifest.json"
        write_json(condition_manifest_path, condition_manifest)
        write_jsonl(condition_dir / "sample_preview.jsonl", previews)
        condition_entries[condition] = {
            "manifest_path": str(condition_manifest_path.relative_to(root)).replace("\\", "/"),
            "manifest_sha256": sha256_file(condition_manifest_path),
        }

    manifest = {
        "schema_version": "tcmi_dataset_manifest_v1",
        "dataset_schema_version": data_config["schema_version"],
        "created_at": datetime.now(UTC).isoformat(),
        "evidence_level": config["project"]["evidence_level"],
        "data_config_hash": stable_hash(data_config),
        "generation_seed": generation_seed,
        "conditions": condition_entries,
        "latent_digests": reference_latent_digests,
        "vocabulary": VOCABULARY.to_manifest(),
        "projection_contract": projection_contract(),
    }
    write_json(root / "manifest.json", manifest)
    return root


def _graph_batches(
    generation_seed: int,
    split: str,
    split_size: int,
    shard_size: int,
    heldout_modulus: int,
) -> Iterator[tuple[int, list[SceneGraph]]]:
    for start_index in range(0, split_size, shard_size):
        end_index = min(split_size, start_index + shard_size)
        yield (
            start_index,
            [
                sample_graph(generation_seed, split, index, heldout_modulus)
                for index in range(start_index, end_index)
            ],
        )


def _build_shard(
    graphs: list[SceneGraph],
    start_index: int,
    split: str,
    condition: str,
    generation_seed: int,
    heldout_modulus: int,
    image_size: int,
) -> dict[str, np.ndarray]:
    images = []
    input_ids = []
    labels = []
    unseen = []
    latent_fields = []
    sample_ids = []
    for local_index, graph in enumerate(graphs):
        global_index = start_index + local_index
        images.append(render_image(graph, condition, image_size))
        input_ids.append(
            text_token_ids(graph, condition, split, generation_seed, global_index)
        )
        labels.append(labels_as_tuple(graph))
        unseen.append(is_unseen_combination(graph, heldout_modulus))
        latent_fields.append(graph.latent_tuple())
        sample_ids.append(f"{split}-{global_index:08d}")
    return {
        "images": np.stack(images).astype(np.uint8),
        "input_ids": np.asarray(input_ids, dtype=np.int16),
        "attention_mask": np.ones((len(graphs), 14), dtype=np.bool_),
        "labels": np.asarray(labels, dtype=np.int16),
        "is_unseen_combination": np.asarray(unseen, dtype=np.bool_),
        "latent_fields": np.asarray(latent_fields, dtype=np.int64),
        "sample_ids": np.asarray(sample_ids),
    }


def projection_contract() -> dict[str, Any]:
    return {
        "redundant": {
            "image": {
                "entity": True,
                "predicate": True,
                "direction": True,
                "joint_graph": True,
            },
            "text": {
                "entity": True,
                "predicate": True,
                "direction": True,
                "joint_graph": True,
            },
            "joint": {
                "entity": True,
                "predicate": True,
                "direction": True,
                "joint_graph": True,
            },
        },
        "complementary": {
            "alignment_anchor": ["predicate", "direction"],
            "factor_visibility": {
                "entity": {
                    "image": ["subject_shape"],
                    "text": ["object_shape"],
                    "joint": ["subject_shape", "object_shape"],
                },
                "joint_graph": {
                    "image": ["visual_key"],
                    "text": ["text_key"],
                    "joint": ["visual_key", "text_key"],
                },
            },
            "image": {
                "entity": False,
                "predicate": True,
                "direction": True,
                "joint_graph": False,
            },
            "text": {
                "entity": False,
                "predicate": True,
                "direction": True,
                "joint_graph": False,
            },
            "joint": {
                "entity": True,
                "predicate": True,
                "direction": True,
                "joint_graph": True,
            },
        },
        "irrelevant": {
            "image": {
                "entity": True,
                "predicate": True,
                "direction": True,
                "joint_graph": True,
            },
            "text": {
                "entity": False,
                "predicate": False,
                "direction": False,
                "joint_graph": False,
            },
            "joint": {
                "entity": True,
                "predicate": True,
                "direction": True,
                "joint_graph": True,
            },
        },
        "conflict": {
            "image": {
                "entity": True,
                "predicate": True,
                "direction": True,
                "joint_graph": True,
            },
            "text_train": {
                "entity": True,
                "predicate": True,
                "direction": True,
                "joint_graph": True,
            },
            "text_test_reliability": "systematically_flipped",
            "joint_test_reliability": "contains_conflicting_text",
        },
    }
