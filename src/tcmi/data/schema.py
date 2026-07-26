from __future__ import annotations

from dataclasses import asdict, dataclass

from tcmi.constants import TASKS


@dataclass(frozen=True)
class SceneGraph:
    subject_shape: int
    object_shape: int
    predicate: int
    direction: int
    visual_key: int
    text_key: int
    layout_jitter_x: int
    layout_jitter_y: int
    template_id: int
    scene_nonce: int

    @property
    def entity_label(self) -> int:
        return self.subject_shape * 4 + self.object_shape

    @property
    def joint_graph_label(self) -> int:
        return self.visual_key ^ self.text_key

    @property
    def labels(self) -> dict[str, int]:
        return {
            "entity": self.entity_label,
            "predicate": self.predicate,
            "direction": self.direction,
            "joint_graph": self.joint_graph_label,
        }

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def latent_tuple(self) -> tuple[int, ...]:
        values = self.to_dict()
        return tuple(values[key] for key in values)


LABEL_COLUMNS = TASKS


def labels_as_tuple(graph: SceneGraph) -> tuple[int, ...]:
    labels = graph.labels
    return tuple(labels[task] for task in LABEL_COLUMNS)
