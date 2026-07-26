from __future__ import annotations

from dataclasses import dataclass

SLOT_NAMES = (
    "subject",
    "object",
    "predicate",
    "direction",
    "visual_key",
    "text_key",
)


def _build_tokens() -> tuple[str, ...]:
    tokens = ["<pad>", "<bos>", "<eos>", "<mask>"]
    tokens.extend(SLOT_NAMES)
    tokens.extend(f"shape_{index}" for index in range(4))
    tokens.extend(f"predicate_{index}" for index in range(4))
    tokens.extend(f"direction_{index}" for index in range(2))
    tokens.extend(f"visual_key_{index}" for index in range(2))
    tokens.extend(f"text_key_{index}" for index in range(2))
    tokens.extend(f"nuisance_{index}" for index in range(64))
    return tuple(tokens)


TOKENS = _build_tokens()
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...] = TOKENS

    @property
    def pad_id(self) -> int:
        return TOKEN_TO_ID["<pad>"]

    @property
    def bos_id(self) -> int:
        return TOKEN_TO_ID["<bos>"]

    @property
    def eos_id(self) -> int:
        return TOKEN_TO_ID["<eos>"]

    @property
    def mask_id(self) -> int:
        return TOKEN_TO_ID["<mask>"]

    def token_id(self, token: str) -> int:
        return TOKEN_TO_ID[token]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self.tokens[token_id] for token_id in token_ids)

    def to_manifest(self) -> dict[str, object]:
        return {
            "size": len(self.tokens),
            "tokens": list(self.tokens),
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "mask_id": self.mask_id,
        }


VOCABULARY = Vocabulary()
