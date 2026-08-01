from __future__ import annotations

import random
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcmi.constants import ARCHITECTURES, INFORMATION_CONDITIONS, TRAIN_MODES
from tcmi.data.dataset import ShardedSceneGraphDataset, dataset_manifest_hash
from tcmi.data.generator import dataset_root
from tcmi.data.vocab import VOCABULARY
from tcmi.evidence import RunContext, RunIdentity, enforce_evidence_gate
from tcmi.io import read_json, write_json
from tcmi.models import build_model, parameter_audit
from tcmi.training.budget import BudgetTracker
from tcmi.training.checkpoint import save_checkpoint
from tcmi.training.losses import symmetric_contrastive_loss


class TrainingError(RuntimeError):
    """Raised when a run violates the training contract."""


def train_run(
    config: dict[str, Any],
    architecture: str,
    train_mode: str,
    condition: str,
    seed: int,
) -> Path:
    _validate_identity(architecture, train_mode, condition)
    evidence_level = enforce_evidence_gate(config)
    _require_passed_data_audit(config)
    set_reproducibility(seed, bool(config["training"]["deterministic_algorithms"]))
    device = resolve_device(str(config["training"]["device"]))

    manifest_hash = dataset_manifest_hash(config)
    identity = RunIdentity(
        architecture=architecture,
        train_mode=train_mode,
        condition=condition,
        seed=seed,
        evidence_level=evidence_level,
    )
    context = RunContext(config, identity, manifest_hash)
    model = build_model(config, architecture)
    _configure_trainable_parameters(model, train_mode)
    model.to(device)
    audit = parameter_audit(model)
    write_json(context.run_dir / "parameter_audit.json", audit)

    train_dataset = ShardedSceneGraphDataset(config, condition, "train")
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        drop_last=True,
        generator=loader_generator,
        persistent_workers=int(config["training"]["num_workers"]) > 0,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    tracker = BudgetTracker(device)
    previously_consumed_seconds = _completed_budget_seconds(config)
    _enforce_gpu_budget(config, previously_consumed_seconds)
    tracker.start()

    try:
        epochs = int(config["training"]["epochs"])
        checkpoint_every = int(config["training"]["checkpoint_every_epochs"])
        for epoch in range(1, epochs + 1):
            epoch_loss = _train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device=device,
                config=config,
                train_mode=train_mode,
                tracker=tracker,
            )
            context.log("epoch_completed", epoch=epoch, mean_loss=epoch_loss)
            _enforce_gpu_budget(
                config,
                previously_consumed_seconds + tracker.elapsed_wall_seconds(),
            )
            if epoch % checkpoint_every == 0 and epoch != epochs:
                save_checkpoint(
                    context.run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config_hash=config["_meta"]["config_hash"],
                    dataset_manifest_hash=manifest_hash,
                    run_identity=identity.__dict__,
                    budget=tracker.to_dict(),
                )
        tracker.stop()
        budget = tracker.to_dict()
        save_checkpoint(
            context.run_dir / "checkpoints" / f"epoch_{epochs:04d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epochs,
            config_hash=config["_meta"]["config_hash"],
            dataset_manifest_hash=manifest_hash,
            run_identity=identity.__dict__,
            budget=budget,
        )
        write_json(context.run_dir / "budget.json", budget)
        context.update_status(
            "completed",
            completed_epochs=epochs,
            budget=budget,
            final_checkpoint=f"checkpoints/epoch_{epochs:04d}.pt",
        )
    except Exception as error:
        budget_stop_error = None
        try:
            tracker.stop()
        except RuntimeError as stop_error:
            budget_stop_error = str(stop_error)
        write_json(context.run_dir / "budget.json", tracker.to_dict())
        context.log(
            "run_failed",
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            budget_stop_error=budget_stop_error,
        )
        context.update_status("failed", failure_type=type(error).__name__)
        raise
    return context.run_dir


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict[str, Any],
    train_mode: str,
    tracker: BudgetTracker,
) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, image_count, text_count = _mode_loss(
            model=model,
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            train_mode=train_mode,
            image_noise_std=float(config["training"]["image_noise_std"]),
            text_dropout_probability=float(
                config["training"]["text_dropout_probability"]
            ),
        )
        loss.backward()
        optimizer.step()
        valid_text_tokens = int(attention_mask.sum().item()) * text_count
        tracker.record_batch(
            batch_size=images.shape[0],
            image_encodings_per_sample=image_count,
            text_sequences_per_sample=text_count,
            valid_text_tokens=valid_text_tokens,
        )
        total_loss += float(loss.detach().cpu())
        batch_count += 1
    if batch_count == 0:
        raise TrainingError("训练 DataLoader 没有产生 batch")
    return total_loss / batch_count


def _mode_loss(
    *,
    model: Any,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    train_mode: str,
    image_noise_std: float,
    text_dropout_probability: float,
) -> tuple[torch.Tensor, int, int]:
    temperature = model.temperature
    if train_mode in {"image_only", "image_only_matched"}:
        image_batch = images
        midpoint = images.shape[0]
        if train_mode == "image_only_matched":
            if images.shape[0] < 4 or images.shape[0] % 2:
                raise TrainingError(
                    "image_only_matched 要求偶数且至少为 4 的 batch size"
                )
            midpoint = images.shape[0] // 2
            image_batch = images[:midpoint]
        first_image = _augment_images(image_batch, image_noise_std)
        second_image = _augment_images(image_batch, image_noise_std)
        image_loss = symmetric_contrastive_loss(
            model.encode_image(first_image),
            model.encode_image(second_image),
            temperature,
        )
        if train_mode == "image_only":
            return image_loss, 2, 0
        text_batch = input_ids[midpoint:]
        text_mask = attention_mask[midpoint:]
        first_text = _augment_text(text_batch, text_dropout_probability)
        second_text = _augment_text(text_batch, text_dropout_probability)
        text_loss = symmetric_contrastive_loss(
            model.encode_text(first_text, text_mask),
            model.encode_text(second_text, text_mask),
            temperature,
        )
        return 0.5 * (image_loss + text_loss), 1, 1

    if train_mode == "text_only":
        first_text = _augment_text(input_ids, text_dropout_probability)
        second_text = _augment_text(input_ids, text_dropout_probability)
        loss = symmetric_contrastive_loss(
            model.encode_text(first_text, attention_mask),
            model.encode_text(second_text, attention_mask),
            temperature,
        )
        return loss, 0, 2

    image_features = model.encode_image(images)
    if train_mode == "multimodal_shuffled":
        shift = int(torch.randint(1, images.shape[0], (1,), device=images.device).item())
        input_ids = input_ids.roll(shifts=shift, dims=0)
        attention_mask = attention_mask.roll(shifts=shift, dims=0)
    text_features = model.encode_text(input_ids, attention_mask)
    loss = symmetric_contrastive_loss(image_features, text_features, temperature)
    return loss, 1, 1


def _augment_images(images: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0:
        return images
    noise = torch.randn(images.shape, device=images.device, dtype=images.dtype) * noise_std
    return (images + noise).clamp(0, 1)


def _augment_text(input_ids: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return input_ids
    augmented = input_ids.clone()
    value_positions = torch.tensor([2, 4, 6, 8, 10, 12], device=input_ids.device)
    dropout_mask = torch.rand(
        (input_ids.shape[0], len(value_positions)),
        device=input_ids.device,
    ) < probability
    values = augmented[:, value_positions]
    values[dropout_mask] = VOCABULARY.mask_id
    augmented[:, value_positions] = values
    return augmented


def _configure_trainable_parameters(model: Any, train_mode: str) -> None:
    if train_mode == "image_only":
        for parameter in model.text_encoder.parameters():
            parameter.requires_grad = False
        for parameter in model.text_projection.parameters():
            parameter.requires_grad = False
    elif train_mode == "text_only":
        for parameter in model.image_encoder.parameters():
            parameter.requires_grad = False
        for parameter in model.image_projection.parameters():
            parameter.requires_grad = False


def _validate_identity(architecture: str, train_mode: str, condition: str) -> None:
    if architecture not in ARCHITECTURES:
        raise TrainingError(f"未知 architecture: {architecture}")
    if train_mode not in TRAIN_MODES:
        raise TrainingError(f"未知 train_mode: {train_mode}")
    if condition not in INFORMATION_CONDITIONS:
        raise TrainingError(f"未知 condition: {condition}")
    if train_mode == "multimodal_conflict" and condition != "conflict":
        raise TrainingError("multimodal_conflict 必须使用 conflict 数据条件")
    if train_mode == "multimodal_aligned" and condition == "conflict":
        raise TrainingError(
            "conflict 数据条件必须使用 multimodal_conflict，避免重复同构运行"
        )


def _require_passed_data_audit(config: dict[str, Any]) -> None:
    report_path = dataset_root(config) / "audit_report.json"
    if not report_path.exists():
        raise TrainingError("训练前必须先运行数据审计")
    report = read_json(report_path)
    if report.get("status") != "passed":
        raise TrainingError("数据审计未通过，禁止训练")


def set_reproducibility(seed: int, deterministic_algorithms: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=False)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingError("配置要求 CUDA，但当前不可用")
    return device


def _completed_budget_seconds(config: dict[str, Any]) -> float:
    from tcmi.config import output_root

    total = 0.0
    budget_paths = list(output_root(config).glob("runs/*/budget.json"))
    budget_paths.extend(output_root(config).glob("runs/*/probes/*/probe_budget.json"))
    for path in budget_paths:
        budget = read_json(path)
        seconds = budget.get("gpu_time_seconds")
        if seconds is None:
            seconds = budget.get("wall_clock_seconds", 0.0)
        total += float(seconds)
    return total


def _enforce_gpu_budget(config: dict[str, Any], consumed_seconds: float) -> None:
    budget_hours = float(config["project"]["gpu_hour_budget"])
    if consumed_seconds >= budget_hours * 3600:
        raise TrainingError(
            f"累计运行时间已达到 {budget_hours:g} GPU 小时预算，停止后续 epoch"
        )
