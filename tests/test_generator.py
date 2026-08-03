from __future__ import annotations

import numpy as np

from tcmi.constants import PRIMARY_SCOPE_BY_TASK, TASK_CLASS_COUNTS
from tcmi.data.generator import (
    complementary_object_shape,
    generate_dataset,
    is_unseen_combination,
    projection_contract,
    render_image,
    sample_graph,
    text_token_ids,
)
from tcmi.data.schema import SceneGraph
from tcmi.data.vocab import VOCABULARY


def test_scene_graph_sampling_is_deterministic() -> None:
    first = sample_graph(20260726, "train", 7, heldout_modulus=4)
    second = sample_graph(20260726, "train", 7, heldout_modulus=4)
    assert first == second


def test_latent_graph_is_shared_but_projection_changes() -> None:
    graph = sample_graph(20260726, "test", 3, heldout_modulus=4)
    redundant_image = render_image(graph, "redundant", 32)
    complementary_image = render_image(graph, "complementary", 32)
    assert redundant_image.shape == (3, 32, 32)
    assert complementary_image.shape == (3, 32, 32)
    assert not np.array_equal(redundant_image, complementary_image)

    redundant_text = text_token_ids(graph, "redundant", "test", 20260726, 3)
    irrelevant_text = text_token_ids(graph, "irrelevant", "test", 20260726, 3)
    assert len(redundant_text) == 14
    assert len(irrelevant_text) == 14
    assert redundant_text != irrelevant_text
    assert redundant_text[0] == VOCABULARY.bos_id
    assert redundant_text[-1] == VOCABULARY.eos_id


def test_conflict_is_aligned_in_train_and_flipped_in_test() -> None:
    train_graph = sample_graph(20260726, "train", 4, heldout_modulus=4)
    test_graph = sample_graph(20260726, "test", 4, heldout_modulus=4)
    assert text_token_ids(
        train_graph, "conflict", "train", 20260726, 4
    ) == text_token_ids(train_graph, "redundant", "train", 20260726, 4)
    assert text_token_ids(
        test_graph, "conflict", "test", 20260726, 4
    ) != text_token_ids(test_graph, "redundant", "test", 20260726, 4)


def test_complementary_entity_and_joint_graph_are_split_across_modalities() -> None:
    graph = SceneGraph(
        subject_shape=1,
        object_shape=3,
        predicate=2,
        direction=0,
        visual_key=1,
        text_key=0,
        layout_jitter_x=0,
        layout_jitter_y=0,
        template_id=0,
        scene_nonce=5,
    )
    assert complementary_object_shape(graph) == 1
    assert graph.joint_graph_label == 2
    assert TASK_CLASS_COUNTS["joint_graph"] == 4
    assert PRIMARY_SCOPE_BY_TASK["entity"] == "fused"
    assert PRIMARY_SCOPE_BY_TASK["joint_graph"] == "fused"

    complementary_text = text_token_ids(
        graph,
        "complementary",
        "test",
        20260726,
        3,
    )
    assert complementary_text[4] == VOCABULARY.token_id("shape_3")
    assert complementary_text[6] == VOCABULARY.token_id("predicate_2")
    assert complementary_text[8] == VOCABULARY.token_id("direction_0")
    assert complementary_text[12] == VOCABULARY.token_id("text_key_0")
    assert complementary_text[2] != VOCABULARY.token_id("shape_1")

    contract = projection_contract()["complementary"]
    assert contract["image"]["entity"] is False
    assert contract["text"]["entity"] is False
    assert contract["text"]["predicate"] is True
    assert contract["text"]["direction"] is True
    assert contract["joint"]["entity"] is True
    assert contract["joint"]["joint_graph"] is True
    assert contract["factor_visibility"]["entity"] == {
        "image": ["subject_shape"],
        "text": ["object_shape"],
        "joint": ["subject_shape", "object_shape"],
    }
    assert contract["alignment_anchor"] == ["predicate", "direction"]


def test_heldout_rule_is_marginally_balanced() -> None:
    def make_graph(subject: int, obj: int, predicate: int) -> SceneGraph:
        return SceneGraph(
            subject_shape=subject,
            object_shape=obj,
            predicate=predicate,
            direction=0,
            visual_key=0,
            text_key=0,
            layout_jitter_x=0,
            layout_jitter_y=0,
            template_id=0,
            scene_nonce=0,
        )

    for subject in range(4):
        for obj in range(4):
            excluded_predicates = [
                predicate
                for predicate in range(4)
                if is_unseen_combination(make_graph(subject, obj, predicate), 4)
            ]
            assert len(excluded_predicates) == 1
        for predicate in range(4):
            excluded_objects = [
                obj
                for obj in range(4)
                if is_unseen_combination(make_graph(subject, obj, predicate), 4)
            ]
            assert len(excluded_objects) == 1
    for obj in range(4):
        for predicate in range(4):
            excluded_subjects = [
                subject
                for subject in range(4)
                if is_unseen_combination(make_graph(subject, obj, predicate), 4)
            ]
            assert len(excluded_subjects) == 1


def test_small_dataset_generation_produces_all_manifests(
    tiny_temp_config: dict,
) -> None:
    root = generate_dataset(tiny_temp_config)
    assert (root / "manifest.json").is_file()
    for condition in tiny_temp_config["data"]["conditions"]:
        assert (root / condition / "manifest.json").is_file()
        assert (root / condition / "sample_preview.jsonl").is_file()
