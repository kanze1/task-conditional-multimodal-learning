from __future__ import annotations

import torch
from torch.nn import functional as F


def symmetric_contrastive_loss(
    left_features: torch.Tensor,
    right_features: torch.Tensor,
    temperature: torch.Tensor | float,
) -> torch.Tensor:
    if left_features.shape != right_features.shape:
        raise ValueError("对比学习两侧 feature shape 必须一致")
    if left_features.shape[0] < 2:
        raise ValueError("对比学习 batch size 必须至少为 2")
    logits = left_features @ right_features.transpose(0, 1)
    logits = logits / torch.as_tensor(
        temperature,
        dtype=logits.dtype,
        device=logits.device,
    ).clamp_min(1e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )
