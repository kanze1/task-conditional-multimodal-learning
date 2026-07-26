from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class BudgetCounters:
    samples: int = 0
    optimizer_updates: int = 0
    image_encodings: int = 0
    text_sequences: int = 0
    text_tokens: int = 0
    wall_clock_seconds: float = 0.0
    gpu_time_seconds: float | None = None


class BudgetTracker:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.counters = BudgetCounters()
        self._wall_start: float | None = None
        self._gpu_start: torch.cuda.Event | None = None
        self._gpu_end: torch.cuda.Event | None = None

    def start(self) -> None:
        self._wall_start = time.perf_counter()
        if self.device.type == "cuda":
            self._gpu_start = torch.cuda.Event(enable_timing=True)
            self._gpu_end = torch.cuda.Event(enable_timing=True)
            self._gpu_start.record()

    def record_batch(
        self,
        batch_size: int,
        image_encodings_per_sample: int,
        text_sequences_per_sample: int,
        valid_text_tokens: int,
    ) -> None:
        self.counters.samples += batch_size
        self.counters.optimizer_updates += 1
        self.counters.image_encodings += batch_size * image_encodings_per_sample
        self.counters.text_sequences += batch_size * text_sequences_per_sample
        self.counters.text_tokens += valid_text_tokens

    def stop(self) -> None:
        if self._wall_start is None:
            raise RuntimeError("BudgetTracker.start 尚未调用")
        self.counters.wall_clock_seconds = time.perf_counter() - self._wall_start
        if self._gpu_end is not None and self._gpu_start is not None:
            self._gpu_end.record()
            torch.cuda.synchronize(self.device)
            self.counters.gpu_time_seconds = (
                self._gpu_start.elapsed_time(self._gpu_end) / 1000.0
            )

    def elapsed_wall_seconds(self) -> float:
        if self._wall_start is None:
            return 0.0
        return time.perf_counter() - self._wall_start

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self.counters)
        if self._wall_start is not None and payload["wall_clock_seconds"] == 0:
            payload["wall_clock_seconds"] = self.elapsed_wall_seconds()
        return payload
