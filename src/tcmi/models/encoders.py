from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from tcmi.data.vocab import TOKENS


class TinyViTImageEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        width: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        patch_size = 4
        if image_size % patch_size:
            raise ValueError("image_size 必须能被 patch_size=4 整除")
        self.patch_embed = nn.Conv2d(3, width, patch_size, stride=patch_size)
        patch_count = (image_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position_embedding = nn.Parameter(torch.zeros(1, patch_count + 1, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            # norm_first 使 nested tensor 路径本就不启用；显式关闭以免每次
            # 构造模型时向 stderr 发出 UserWarning。
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(width)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((class_tokens, patches), dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        encoded = self.transformer(tokens)
        return self.norm(encoded[:, 0])


class TinyCNNImageEncoder(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        channels = (width // 2, width, width * 2)
        blocks: list[nn.Module] = []
        input_channels = 3
        for output_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], width),
            nn.LayerNorm(width),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.output(self.pool(self.features(images)))


class TextTransformerEncoder(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        dropout: float,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(len(TOKENS), width)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_sequence_length, width)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            # norm_first 使 nested tensor 路径本就不启用；显式关闭以免每次
            # 构造模型时向 stderr 发出 UserWarning。
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(width)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.token_embedding(input_ids)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        padding_mask = ~attention_mask.bool()
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        weights = attention_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.norm(pooled)


class DualTower(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        image_width: int,
        text_width: int,
        embedding_dim: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_projection = nn.Linear(image_width, embedding_dim, bias=False)
        self.text_projection = nn.Linear(text_width, embedding_dim, bias=False)
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_projection(self.image_encoder(images))
        return F.normalize(features, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        features = self.text_projection(self.text_encoder(input_ids, attention_mask))
        return F.normalize(features, dim=-1)


def build_model(config: dict[str, Any], architecture: str) -> DualTower:
    data_config = config["data"]
    model_config = config["model"]
    image_width = int(model_config["image_width"])
    text_width = int(model_config["text_width"])
    layers = int(model_config["transformer_layers"])
    heads = int(model_config["attention_heads"])
    dropout = float(model_config["dropout"])

    if architecture == "tiny_vit":
        image_encoder: nn.Module = TinyViTImageEncoder(
            image_size=int(data_config["image_size"]),
            width=image_width,
            layers=layers,
            heads=heads,
            dropout=dropout,
        )
    elif architecture == "tiny_cnn":
        image_encoder = TinyCNNImageEncoder(width=image_width, dropout=dropout)
    else:
        raise ValueError(f"未知 architecture: {architecture}")

    text_encoder = TextTransformerEncoder(
        width=text_width,
        layers=layers,
        heads=heads,
        dropout=dropout,
        max_sequence_length=int(data_config["max_seq_length"]),
    )
    model = DualTower(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        image_width=image_width,
        text_width=text_width,
        embedding_dim=int(model_config["embedding_dim"]),
        temperature=float(model_config["temperature"]),
    )
    _initialize_linear_layers(model)
    return model


def _initialize_linear_layers(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.trunc_normal_(child.weight, std=1 / math.sqrt(child.in_features))
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def parameter_audit(model: DualTower) -> dict[str, int]:
    def count(module: nn.Module, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in module.parameters()
            if not trainable_only or parameter.requires_grad
        )

    return {
        "total_parameters": count(model),
        "trainable_parameters": count(model, trainable_only=True),
        "image_encoder_parameters": count(model.image_encoder),
        "text_encoder_parameters": count(model.text_encoder),
        "image_projection_parameters": count(model.image_projection),
        "text_projection_parameters": count(model.text_projection),
    }
