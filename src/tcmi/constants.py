from __future__ import annotations

INFORMATION_CONDITIONS = (
    "redundant",
    "complementary",
    "irrelevant",
    "conflict",
)

TRAIN_MODES = (
    "image_only",
    "image_only_matched",
    "text_only",
    "multimodal_aligned",
    "multimodal_shuffled",
    "multimodal_conflict",
)

ARCHITECTURES = ("tiny_vit", "tiny_cnn")
REPRESENTATION_SCOPES = ("image", "text", "fused")
EVIDENCE_LEVELS = ("smoke", "pilot", "formal")
SPLITS = ("train", "validation", "test")
TASKS = ("entity", "predicate", "direction", "joint_graph")

TASK_CLASS_COUNTS = {
    "entity": 16,
    "predicate": 4,
    "direction": 2,
    "joint_graph": 2,
}
