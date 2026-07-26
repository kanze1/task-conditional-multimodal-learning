from __future__ import annotations

import numpy as np

from tcmi.data.generator import (
    generate_dataset,
    render_image,
    sample_graph,
    text_token_ids,
)
from tcmi.data.vocab import VOCABULARY


def test_scene_graph_sampling_is_deterministic() -> None:
    first = sample_graph(20260726, "train", 7, heldout_modulus=5)
    second = sample_graph(20260726, "train", 7, heldout_modulus=5)
    assert first == second


def test_latent_graph_is_shared_but_projection_changes() -> None:
    graph = sample_graph(20260726, "test", 3, heldout_modulus=5)
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
    train_graph = sample_graph(20260726, "train", 4, heldout_modulus=5)
    test_graph = sample_graph(20260726, "test", 4, heldout_modulus=5)
    assert text_token_ids(
        train_graph, "conflict", "train", 20260726, 4
    ) == text_token_ids(train_graph, "redundant", "train", 20260726, 4)
    assert text_token_ids(
        test_graph, "conflict", "test", 20260726, 4
    ) != text_token_ids(test_graph, "redundant", "test", 20260726, 4)

def test_small_dataset_generation_produces_all_manifests(
    tiny_temp_config: dict,
) -> None:
    root = generate_dataset(tiny_temp_config)
    assert (root / "manifest.json").is_file()
    for condition in tiny_temp_config["data"]["conditions"]:
        assert (root / condition / "manifest.json").is_file()
        assert (root / condition / "sample_preview.jsonl").is_file()
