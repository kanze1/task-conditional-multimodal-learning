from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tcmi.config import canonical_json
from tcmi.constants import INFORMATION_CONDITIONS, SPLITS
from tcmi.data.generator import dataset_root, projection_contract, text_token_ids
from tcmi.data.oracle import run_oracle_audit
from tcmi.data.schema import SceneGraph
from tcmi.io import read_json, sha256_file, write_json


class AuditFailure(RuntimeError):
    """Raised when generated data violates a preregistered invariant."""


def audit_dataset(config: dict[str, Any]) -> Path:
    root = dataset_root(config)
    root_manifest_path = root / "manifest.json"
    root_manifest = read_json(root_manifest_path)
    failures: list[str] = []
    condition_summaries: dict[str, Any] = {}
    latent_reference: dict[str, str] | None = None
    sample_ids_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    latent_keys_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}

    for condition in INFORMATION_CONDITIONS:
        entry = root_manifest["conditions"].get(condition)
        if entry is None:
            failures.append(f"缺少条件 manifest: {condition}")
            continue
        condition_manifest_path = root / entry["manifest_path"]
        if sha256_file(condition_manifest_path) != entry["manifest_sha256"]:
            failures.append(f"条件 manifest 哈希不匹配: {condition}")
        condition_manifest = read_json(condition_manifest_path)
        if latent_reference is None:
            latent_reference = condition_manifest["latent_digests"]
        elif condition_manifest["latent_digests"] != latent_reference:
            failures.append(f"潜在图摘要跨条件不一致: {condition}")

        condition_summaries[condition] = {}
        for split in SPLITS:
            split_summary = _audit_split(
                root=root,
                condition=condition,
                split=split,
                split_manifest=condition_manifest["splits"][split],
                config=config,
                failures=failures,
                reference_sample_ids=sample_ids_by_split[split],
                reference_latent_keys=latent_keys_by_split[split],
            )
            condition_summaries[condition][split] = split_summary

    split_names = list(SPLITS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = sample_ids_by_split[left] & sample_ids_by_split[right]
            if overlap:
                failures.append(f"split sample_id 重叠: {left}/{right}, count={len(overlap)}")
            latent_overlap = latent_keys_by_split[left] & latent_keys_by_split[right]
            if latent_overlap:
                failures.append(
                    f"split 潜在样本重复: {left}/{right}, count={len(latent_overlap)}"
                )

    oracle_report = run_oracle_audit(config)
    failures.extend(oracle_report["failures"])

    report = {
        "schema_version": "tcmi_data_audit_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_manifest_sha256": sha256_file(root_manifest_path),
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "condition_summaries": condition_summaries,
        "projection_contract": projection_contract(),
        "oracle_audit": oracle_report,
        "checks": {
            "manifest_hashes": True,
            "cross_condition_latent_identity": True,
            "sample_id_split_isolation": True,
            "latent_content_deduplication": True,
            "label_invariants": True,
            "text_projection_semantics": True,
            "fixed_sequence_length": True,
            "unseen_combination_partition": True,
            "oracle_identifiability": True,
        },
    }
    report_path = root / "audit_report.json"
    write_json(report_path, report)
    if failures:
        raise AuditFailure(f"数据审计失败，详见 {report_path}")
    return report_path


def _audit_split(
    root: Path,
    condition: str,
    split: str,
    split_manifest: dict[str, Any],
    config: dict[str, Any],
    failures: list[str],
    reference_sample_ids: set[str],
    reference_latent_keys: set[str],
) -> dict[str, Any]:
    labels = Counter()
    unseen_count = 0
    seen_count = 0
    observed_count = 0
    latent_digest = hashlib.sha256()
    current_sample_ids: set[str] = set()
    current_latent_keys: set[str] = set()

    for shard in split_manifest["shards"]:
        shard_path = root / shard["path"]
        if sha256_file(shard_path) != shard["sha256"]:
            failures.append(f"shard 哈希不匹配: {shard['path']}")
            continue
        with np.load(shard_path, allow_pickle=False) as arrays:
            shard_count = len(arrays["sample_ids"])
            observed_count += shard_count
            if arrays["input_ids"].shape[1] != int(config["data"]["max_seq_length"]):
                failures.append(f"序列长度错误: {shard['path']}")

            for row_index in range(shard_count):
                sample_id = str(arrays["sample_ids"][row_index])
                if sample_id in current_sample_ids:
                    failures.append(f"split 内 sample_id 重复: {sample_id}")
                current_sample_ids.add(sample_id)
                latent = arrays["latent_fields"][row_index].tolist()
                graph = SceneGraph(*[int(value) for value in latent])
                latent_key = canonical_json(graph.to_dict())
                if latent_key in current_latent_keys:
                    failures.append(f"split 内潜在样本重复: {condition}/{sample_id}")
                current_latent_keys.add(latent_key)
                latent_digest.update(
                    (latent_key + "\n").encode("utf-8")
                )
                expected_labels = [
                    graph.entity_label,
                    graph.predicate,
                    graph.direction,
                    graph.joint_graph_label,
                ]
                observed_labels = arrays["labels"][row_index].astype(int).tolist()
                if observed_labels != expected_labels:
                    failures.append(f"标签不变量失败: {sample_id}")
                for task_index, value in enumerate(observed_labels):
                    labels[(task_index, value)] += 1

                expected_ids = text_token_ids(
                    graph=graph,
                    condition=condition,
                    split=split,
                    generation_seed=int(config["data"]["generation_seed"]),
                    index=int(sample_id.rsplit("-", 1)[1]),
                )
                if arrays["input_ids"][row_index].astype(int).tolist() != expected_ids:
                    failures.append(f"文本投影不匹配: {condition}/{sample_id}")

                if bool(arrays["is_unseen_combination"][row_index]):
                    unseen_count += 1
                else:
                    seen_count += 1

    if observed_count != int(split_manifest["count"]):
        failures.append(
            f"split 计数不匹配: {condition}/{split}, "
            f"expected={split_manifest['count']}, observed={observed_count}"
        )
    if condition == INFORMATION_CONDITIONS[0]:
        reference_sample_ids.update(current_sample_ids)
        reference_latent_keys.update(current_latent_keys)
    elif current_sample_ids != reference_sample_ids:
        failures.append(f"sample_id 跨条件不一致: {condition}/{split}")
    elif current_latent_keys != reference_latent_keys:
        failures.append(f"潜在样本集合跨条件不一致: {condition}/{split}")
    if split == "train" and unseen_count != 0:
        failures.append(f"训练集包含 held-out 组合: {condition}")
    if split != "train" and abs(unseen_count - seen_count) > 1:
        failures.append(f"{split} 未保持 seen/unseen 近似平衡: {condition}")

    return {
        "count": observed_count,
        "seen_count": seen_count,
        "unseen_count": unseen_count,
        "latent_digest": latent_digest.hexdigest(),
        "label_histogram": {
            f"task_{task_index}_class_{value}": count
            for (task_index, value), count in sorted(labels.items())
        },
    }
