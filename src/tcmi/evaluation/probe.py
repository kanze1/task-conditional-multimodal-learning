from __future__ import annotations

import copy
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tcmi.config import project_path, public_config, stable_hash
from tcmi.constants import REPRESENTATION_SCOPES, TASK_CLASS_COUNTS, TASKS
from tcmi.data.dataset import dataset_manifest_hash
from tcmi.evaluation.features import (
    FeatureSet,
    checkpoint_provenance,
    extract_features,
    load_model_for_run,
)
from tcmi.evidence import source_snapshot, utc_now
from tcmi.io import read_json, sha256_file, write_json, write_jsonl
from tcmi.training.budget import BudgetTracker
from tcmi.training.trainer import (
    _completed_budget_seconds,
    _enforce_gpu_budget,
    resolve_device,
)


class ProbeError(RuntimeError):
    """Raised when frozen-probe evaluation is incomplete or inconsistent."""


def _probe_provenance(
    config: dict[str, Any],
    directory: Path,
    representation_scope: str,
) -> dict[str, Any]:
    training_manifest = read_json(directory / "run_manifest.json")
    config_hash = stable_hash(public_config(config))
    current_dataset_hash = dataset_manifest_hash(config)
    if training_manifest["config_hash"] != config_hash:
        raise ProbeError("当前配置与训练运行的 config_hash 不一致")
    if training_manifest["dataset_manifest_hash"] != current_dataset_hash:
        raise ProbeError("当前数据 manifest 与训练运行不一致")
    if training_manifest["evidence_level"] != config["project"]["evidence_level"]:
        raise ProbeError("probe 与训练运行的 evidence_level 不一致")

    root = project_path(config)
    source = source_snapshot(root)
    if config["project"]["evidence_level"] == "formal" and source["dirty"]:
        raise ProbeError("formal probe 要求 Git 工作树干净")
    protocol_path = project_path(config, config["project"]["protocol_path"])
    checkpoint_path = directory / training_manifest["final_checkpoint"]
    return {
        "schema_version": "tcmi_probe_run_v1",
        "created_at": utc_now(),
        "evidence_level": config["project"]["evidence_level"],
        "representation_scope": representation_scope,
        "config_hash": config_hash,
        "dataset_manifest_hash": current_dataset_hash,
        "protocol_path": config["project"]["protocol_path"],
        "protocol_sha256": sha256_file(protocol_path),
        "source": source,
        "training_run": {
            "identity": training_manifest["identity"],
            "git_commit": training_manifest["source"]["git_commit"],
            "config_hash": training_manifest["config_hash"],
            "dataset_manifest_hash": training_manifest["dataset_manifest_hash"],
        },
        "checkpoint": checkpoint_provenance(checkpoint_path),
    }


def run_probes(
    config: dict[str, Any],
    run_dir: str | Path,
    representation_scope: str,
) -> Path:
    if representation_scope not in REPRESENTATION_SCOPES:
        raise ProbeError(f"未知 representation_scope: {representation_scope}")
    directory = Path(run_dir)
    output_dir = directory / "probes" / representation_scope
    if output_dir.exists():
        manifest_path = output_dir / "probe_run_manifest.json"
        if manifest_path.is_file():
            existing = read_json(manifest_path)
            if (
                existing.get("status") == "completed"
                and existing.get("config_hash")
                == stable_hash(public_config(config))
            ):
                return output_dir / existing["metrics_path"]
        raise ProbeError(
            f"probe 输出已存在且未完成或配置不一致，拒绝覆盖: {output_dir}"
        )
    provenance = _probe_provenance(config, directory, representation_scope)
    output_dir.mkdir(parents=True)
    device = resolve_device(str(config["training"]["device"]))
    previously_consumed_seconds = _completed_budget_seconds(config)
    _enforce_gpu_budget(config, previously_consumed_seconds)
    tracker = BudgetTracker(device)
    tracker.start()
    write_json(
        output_dir / "probe_run_manifest.json",
        {
            **provenance,
            "status": "running",
        },
    )
    try:
        metrics_path = _run_probes_impl(
            config=config,
            directory=directory,
            representation_scope=representation_scope,
            output_dir=output_dir,
            tracker=tracker,
            previously_consumed_seconds=previously_consumed_seconds,
            probe_provenance=provenance,
        )
        tracker.stop()
        write_json(output_dir / "probe_budget.json", tracker.to_dict())
        write_json(
            output_dir / "probe_run_manifest.json",
            {
                **provenance,
                "status": "completed",
                "updated_at": utc_now(),
                "metrics_path": metrics_path.name,
                "budget_path": "probe_budget.json",
            },
        )
        return metrics_path
    except Exception as error:
        budget_stop_error = None
        try:
            tracker.stop()
        except RuntimeError as stop_error:
            budget_stop_error = str(stop_error)
        write_json(output_dir / "probe_budget.json", tracker.to_dict())
        write_json(
            output_dir / "probe_run_manifest.json",
            {
                **provenance,
                "status": "failed",
                "updated_at": utc_now(),
                "failure_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "budget_stop_error": budget_stop_error,
                "budget_path": "probe_budget.json",
            },
        )
        raise


def _run_probes_impl(
    *,
    config: dict[str, Any],
    directory: Path,
    representation_scope: str,
    output_dir: Path,
    tracker: BudgetTracker,
    previously_consumed_seconds: float,
    probe_provenance: dict[str, Any],
) -> Path:
    model, device, run_manifest, checkpoint_path = load_model_for_run(config, directory)
    identity = run_manifest["identity"]
    condition = identity["condition"]
    seed = int(identity["seed"])

    train_features = extract_features(
        config=config,
        model=model,
        device=device,
        condition=condition,
        split="train",
        representation_scope=representation_scope,
        seed=seed,
    )
    validation_features = extract_features(
        config=config,
        model=model,
        device=device,
        condition=condition,
        split="validation",
        representation_scope=representation_scope,
        seed=seed,
    )
    clean_test = extract_features(
        config=config,
        model=model,
        device=device,
        condition=condition,
        split="test",
        representation_scope=representation_scope,
        seed=seed,
    )
    robustness_sets = _robustness_feature_sets(
        config=config,
        model=model,
        device=device,
        condition=condition,
        representation_scope=representation_scope,
        seed=seed,
    )

    metrics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task_index, task in enumerate(TASKS):
        classifier, validation_accuracy = _fit_linear_probe(
            train=train_features,
            validation=validation_features,
            task_index=task_index,
            class_count=TASK_CLASS_COUNTS[task],
            config=config,
            seed=seed + task_index,
            device=device,
        )
        base_record = {
            **identity,
            "representation_scope": representation_scope,
            "task": task,
            "validation_accuracy": validation_accuracy,
        }
        clean_result = _evaluate_classifier(
            classifier,
            clean_test,
            task_index,
            device,
        )
        metrics.append(
            {
                **base_record,
                "metric": "clean_accuracy",
                "accuracy": clean_result["accuracy"],
                "sample_count": clean_result["sample_count"],
            }
        )
        if condition == "conflict":
            metrics.append(
                {
                    **base_record,
                    "metric": "conflict_flip_accuracy",
                    "accuracy": clean_result["accuracy"],
                    "sample_count": clean_result["sample_count"],
                }
            )
        unseen_result = _evaluate_classifier(
            classifier,
            clean_test,
            task_index,
            device,
            mask=clean_test.unseen_mask,
        )
        metrics.append(
            {
                **base_record,
                "metric": "unseen_combination_accuracy",
                "accuracy": unseen_result["accuracy"],
                "sample_count": unseen_result["sample_count"],
            }
        )
        for metric_name, feature_set in robustness_sets.items():
            result = _evaluate_classifier(
                classifier,
                feature_set,
                task_index,
                device,
            )
            metrics.append(
                {
                    **base_record,
                    "metric": metric_name,
                    "accuracy": result["accuracy"],
                    "sample_count": result["sample_count"],
                }
            )
        failures.extend(
            _failure_examples(
                classifier=classifier,
                feature_set=clean_test,
                task_index=task_index,
                task=task,
                device=device,
                limit=int(config["probe"]["failure_examples"]),
                identity=identity,
                representation_scope=representation_scope,
            )
        )
        _enforce_gpu_budget(
            config,
            previously_consumed_seconds + tracker.elapsed_wall_seconds(),
        )

    payload = {
        "schema_version": "tcmi_probe_metrics_v1",
        "evidence_level": probe_provenance["evidence_level"],
        "run_identity": identity,
        "representation_scope": representation_scope,
        "checkpoint": checkpoint_provenance(checkpoint_path),
        "config_hash": probe_provenance["config_hash"],
        "dataset_manifest_hash": probe_provenance["dataset_manifest_hash"],
        "protocol_sha256": probe_provenance["protocol_sha256"],
        "source": probe_provenance["source"],
        "probe_config": config["probe"],
        "metrics": metrics,
    }
    metrics_path = output_dir / "probe_metrics.json"
    write_json(metrics_path, payload)
    write_jsonl(output_dir / "failure_examples.jsonl", failures)

    run_manifest_path = directory / "run_manifest.json"
    updated_manifest = read_json(run_manifest_path)
    probes = updated_manifest.setdefault("completed_probes", [])
    probes.append(
        {
            "representation_scope": representation_scope,
            "metrics_path": str(metrics_path.relative_to(directory)).replace("\\", "/"),
        }
    )
    write_json(run_manifest_path, updated_manifest)
    return metrics_path


def _fit_linear_probe(
    *,
    train: FeatureSet,
    validation: FeatureSet,
    task_index: int,
    class_count: int,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[nn.Linear, float]:
    torch.manual_seed(seed)
    classifier = nn.Linear(train.features.shape[1], class_count).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=float(config["probe"]["learning_rate"]),
        weight_decay=float(config["probe"]["weight_decay"]),
    )
    train_dataset = TensorDataset(train.features, train.labels[:, task_index])
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["probe"]["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(int(config["probe"]["epochs"])):
        classifier.train()
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(classifier(features), targets)
            loss.backward()
            optimizer.step()
        result = _evaluate_classifier(
            classifier,
            validation,
            task_index,
            device,
        )
        if result["accuracy"] > best_accuracy:
            best_accuracy = result["accuracy"]
            best_state = copy.deepcopy(classifier.state_dict())
    if best_state is None:
        raise ProbeError("线性 probe 未产生有效状态")
    classifier.load_state_dict(best_state)
    classifier.eval()
    return classifier, best_accuracy


@torch.inference_mode()
def _evaluate_classifier(
    classifier: nn.Module,
    feature_set: FeatureSet,
    task_index: int,
    device: torch.device,
    mask: torch.Tensor | None = None,
) -> dict[str, float | int]:
    features = feature_set.features
    targets = feature_set.labels[:, task_index]
    if mask is not None:
        features = features[mask]
        targets = targets[mask]
    if len(features) == 0:
        raise ProbeError("评估子集为空")
    predictions = classifier(features.to(device)).argmax(dim=-1).cpu()
    correct = int((predictions == targets).sum().item())
    return {
        "accuracy": correct / len(targets),
        "sample_count": len(targets),
    }


def _robustness_feature_sets(
    *,
    config: dict[str, Any],
    model: Any,
    device: torch.device,
    condition: str,
    representation_scope: str,
    seed: int,
) -> dict[str, FeatureSet]:
    corruptions = {
        "image": ("noisy_image_accuracy",),
        "text": ("noisy_text_accuracy", "conflicting_text_accuracy"),
        "fused": (
            "noisy_image_accuracy",
            "noisy_text_accuracy",
            "conflicting_text_accuracy",
            "missing_image_accuracy",
            "missing_text_accuracy",
        ),
    }[representation_scope]
    mapping = {
        metric_name: metric_name.removesuffix("_accuracy") for metric_name in corruptions
    }
    return {
        metric_name: extract_features(
            config=config,
            model=model,
            device=device,
            condition=condition,
            split="test",
            representation_scope=representation_scope,
            corruption=corruption,
            seed=seed + index + 1,
        )
        for index, (metric_name, corruption) in enumerate(mapping.items())
    }


@torch.inference_mode()
def _failure_examples(
    *,
    classifier: nn.Module,
    feature_set: FeatureSet,
    task_index: int,
    task: str,
    device: torch.device,
    limit: int,
    identity: dict[str, Any],
    representation_scope: str,
) -> list[dict[str, Any]]:
    logits = classifier(feature_set.features.to(device)).cpu()
    probabilities = logits.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    targets = feature_set.labels[:, task_index]
    wrong_indices = torch.nonzero(predictions != targets).flatten()
    if len(wrong_indices) == 0:
        return []
    wrong_confidences = probabilities[wrong_indices, predictions[wrong_indices]]
    order = torch.argsort(wrong_confidences, descending=True)[:limit]
    selected = wrong_indices[order]
    return [
        {
            **identity,
            "representation_scope": representation_scope,
            "task": task,
            "sample_id": feature_set.sample_ids[int(index)],
            "target": int(targets[index]),
            "prediction": int(predictions[index]),
            "confidence": float(probabilities[index, predictions[index]]),
            "is_unseen_combination": bool(feature_set.unseen_mask[index]),
        }
        for index in selected
    ]
