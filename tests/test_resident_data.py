from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tcmi.evidence as evidence_module
from tcmi.config import ConfigError, public_config, stable_hash, validate_config
from tcmi.data.audit import audit_dataset
from tcmi.data.dataset import ResidentSceneGraphData, ShardedSceneGraphDataset
from tcmi.data.generator import generate_dataset
from tcmi.evaluation.probe import ProbeError, run_probes
from tcmi.io import read_json
from tcmi.training.trainer import TrainingError, train_run


@pytest.fixture
def generated_tiny_config(tiny_temp_config: dict, monkeypatch) -> dict:
    config = tiny_temp_config
    config["training"]["device"] = "cpu"
    config["training"]["batch_size"] = 4
    config["training"]["epochs"] = 1
    config["_meta"]["config_hash"] = stable_hash(public_config(config))
    project_root = Path(config["_meta"]["config_path"]).parent.parent
    protocol_path = project_root / config["project"]["protocol_path"]
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("draft protocol\n", encoding="utf-8")
    monkeypatch.setattr(
        evidence_module,
        "source_snapshot",
        lambda root: {
            "git_commit": "test-commit",
            "dirty": False,
            "dirty_paths": [],
        },
    )
    generate_dataset(config)
    audit_dataset(config)
    return config


def test_resident_data_matches_sharded_dataset(generated_tiny_config: dict) -> None:
    config = generated_tiny_config
    sharded = ShardedSceneGraphDataset(config, "redundant", "train")
    resident = ResidentSceneGraphData(
        config, "redundant", "train", torch.device("cpu")
    )
    assert len(resident) == len(sharded)
    for index in (0, len(sharded) - 1):
        sample = sharded[index]
        assert torch.equal(
            resident.images[index].float() / 255.0, sample["image"]
        )
        assert torch.equal(resident.input_ids[index], sample["input_ids"])
        assert torch.equal(resident.attention_mask[index], sample["attention_mask"])


def test_resident_training_batches_are_seed_deterministic(
    generated_tiny_config: dict,
) -> None:
    config = generated_tiny_config
    resident = ResidentSceneGraphData(
        config, "redundant", "train", torch.device("cpu")
    )

    def collect(seed: int) -> list[torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return [
            batch["input_ids"].clone()
            for batch in resident.training_batches(4, generator)
        ]

    first = collect(2601)
    second = collect(2601)
    different = collect(2602)
    assert len(first) == len(resident) // 4
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert any(
        not torch.equal(a, b) for a, b in zip(first, different, strict=True)
    )


def test_train_run_with_gpu_resident_backend_on_cpu(
    generated_tiny_config: dict,
) -> None:
    config = generated_tiny_config
    config["training"]["data_backend"] = "gpu_resident"
    config["_meta"]["config_hash"] = stable_hash(public_config(config))
    run_dir = train_run(
        config,
        architecture="tiny_cnn",
        train_mode="multimodal_aligned",
        condition="redundant",
        seed=2601,
    )
    manifest = read_json(run_dir / "run_manifest.json")
    assert manifest["status"] == "completed"


def test_train_run_skips_completed_and_rejects_partial(
    generated_tiny_config: dict,
) -> None:
    config = generated_tiny_config
    config["training"]["data_backend"] = "gpu_resident"
    config["_meta"]["config_hash"] = stable_hash(public_config(config))
    identity = {
        "architecture": "tiny_cnn",
        "train_mode": "multimodal_aligned",
        "condition": "redundant",
        "seed": 2601,
    }
    first = train_run(config, **identity)
    resumed = train_run(config, **identity)
    assert resumed == first

    partial = (
        first.parent / "smoke__tiny_cnn__multimodal_aligned__irrelevant__seed_2601"
    )
    partial.mkdir()
    with pytest.raises(TrainingError, match="拒绝覆盖"):
        train_run(
            config,
            architecture="tiny_cnn",
            train_mode="multimodal_aligned",
            condition="irrelevant",
            seed=2601,
        )


def test_run_probes_skips_completed_and_rejects_partial(
    generated_tiny_config: dict,
    monkeypatch,
) -> None:
    import tcmi.evaluation.probe as probe_module

    monkeypatch.setattr(
        probe_module,
        "source_snapshot",
        lambda root: {
            "git_commit": "test-commit",
            "dirty": False,
            "dirty_paths": [],
        },
    )
    config = generated_tiny_config
    config["training"]["data_backend"] = "gpu_resident"
    config["probe"]["epochs"] = 1
    config["probe"]["bootstrap_samples"] = 10
    config["_meta"]["config_hash"] = stable_hash(public_config(config))
    run_dir = train_run(
        config,
        architecture="tiny_cnn",
        train_mode="multimodal_aligned",
        condition="redundant",
        seed=2601,
    )
    first = run_probes(config, run_dir, "image")
    resumed = run_probes(config, run_dir, "image")
    assert resumed == first

    partial_manifest = read_json(
        run_dir / "probes" / "image" / "probe_run_manifest.json"
    )
    partial_manifest["status"] = "failed"
    from tcmi.io import write_json

    write_json(
        run_dir / "probes" / "image" / "probe_run_manifest.json", partial_manifest
    )
    with pytest.raises(ProbeError, match="拒绝覆盖"):
        run_probes(config, run_dir, "image")


def test_config_rejects_unknown_precision_and_backend(smoke_config: dict) -> None:
    config = smoke_config
    config["training"]["precision"] = "fp16"
    with pytest.raises(ConfigError, match="precision"):
        validate_config(config)
    config["training"]["precision"] = "bf16"
    config["training"]["data_backend"] = "mmap"
    with pytest.raises(ConfigError, match="data_backend"):
        validate_config(config)
