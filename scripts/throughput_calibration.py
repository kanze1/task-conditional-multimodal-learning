"""D-MVP-4 要求的短时吞吐校准。

在真实 GPU 上用 pilot 配置的模型规模与 batch 形状测量每种
architecture × train_mode × precision 的训练步速，并按 pilot 矩阵
外推完整训练阶段的 GPU 小时数。数据使用驻留显存的合成张量，
形状与 tcmi_scene_graph_v2 一致，因此测得的是纯训练路径吞吐
（gpu_resident 后端消除了数据管线开销，该外推即为主要预算项）。

用法（项目根目录）：
    python scripts/throughput_calibration.py --config configs/mvp_pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from tcmi.config import load_config
from tcmi.matrix import matrix_cells
from tcmi.models import build_model
from tcmi.training.trainer import _mode_loss, set_reproducibility

MEASURED_MODES = (
    "image_only",
    "image_only_matched",
    "text_only",
    "multimodal_aligned",
)
# shuffled / conflict 的计算图与 aligned 相同，共用其测量值。
MODE_ALIASES = {
    "multimodal_shuffled": "multimodal_aligned",
    "multimodal_conflict": "multimodal_aligned",
}


def measure_steps_per_second(
    config: dict,
    architecture: str,
    train_mode: str,
    precision: str,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
) -> float:
    model = build_model(config, architecture).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch_size = int(config["training"]["batch_size"])
    image_size = int(config["data"]["image_size"])
    seq_length = int(config["data"]["max_seq_length"])
    images = torch.rand(batch_size, 3, image_size, image_size, device=device)
    input_ids = torch.randint(0, 16, (batch_size, seq_length), device=device)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=device)
    autocast_enabled = precision == "bf16" and device.type == "cuda"

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            loss, _, _ = _mode_loss(
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

    for _ in range(warmup_steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(measure_steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return measure_steps / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mvp_pilot.yaml")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--output", default=None, help="结果 JSON 输出路径")
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    if not torch.cuda.is_available():
        raise SystemExit("吞吐校准要求 CUDA GPU")
    device = torch.device("cuda")
    set_reproducibility(2601, bool(config["training"]["deterministic_algorithms"]))

    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    train_samples = int(config["data"]["splits"]["train"])
    steps_per_cell = epochs * (train_samples // batch_size)

    architectures = list(config["matrix"]["architectures"])
    results: dict[str, dict[str, float]] = {}
    for precision in ("fp32", "bf16"):
        torch.backends.cuda.matmul.allow_tf32 = precision == "bf16" and bool(
            config["training"].get("tf32", False)
        )
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        for architecture in architectures:
            for train_mode in MEASURED_MODES:
                key = f"{architecture}/{train_mode}/{precision}"
                steps_per_second = measure_steps_per_second(
                    config,
                    architecture,
                    train_mode,
                    precision,
                    device,
                    arguments.warmup_steps,
                    arguments.measure_steps,
                )
                results[key] = {
                    "steps_per_second": round(steps_per_second, 2),
                    "seconds_per_cell": round(steps_per_cell / steps_per_second, 1),
                }
                print(
                    f"{key}: {steps_per_second:8.2f} steps/s "
                    f"-> {steps_per_cell / steps_per_second:7.1f} s/cell"
                )

    cells = list(matrix_cells(config))
    projection: dict[str, float] = {}
    for precision in ("fp32", "bf16"):
        total_seconds = 0.0
        for cell in cells:
            mode = MODE_ALIASES.get(cell["train_mode"], cell["train_mode"])
            key = f"{cell['architecture']}/{mode}/{precision}"
            total_seconds += results[key]["seconds_per_cell"]
        projection[precision] = round(total_seconds / 3600, 2)

    summary = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "deterministic_algorithms": bool(
            config["training"]["deterministic_algorithms"]
        ),
        "batch_size": batch_size,
        "steps_per_cell": steps_per_cell,
        "matrix_training_cells": len(cells),
        "per_mode_results": results,
        "projected_training_gpu_hours": projection,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if arguments.output:
        output_path = Path(arguments.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"结果已写入 {output_path}")


if __name__ == "__main__":
    main()
