"""Tests for instruction matching and feedback layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rlip.instruction_matching import (
    FeedbackLayer,
    TFIDFEncoder,
    get_feedback_layer,
    record_match_feedback,
    save_feedback_layer,
    score_instruction_against_corpus,
)


def test_tfidf_cosine_baseline_ranks_expected_candidate() -> None:
    instruction = "sail towards the beach shore"
    candidates = [
        ("you are in open deep water far from land", None),
        ("you are near the sandy beach shore", None),
        ("you are at the harbor dock", None),
    ]
    result = score_instruction_against_corpus(instruction, candidates)
    assert result.best_language == "you are near the sandy beach shore"
    assert result.similarity_score == pytest.approx(result.base_similarity_score)
    assert result.similarity_score > 0.0


def test_feedback_layer_boosts_validated_correct_pair() -> None:
    instruction = "go to the red room"
    good = "standing in the red room with a table"
    bad = "standing in the blue hallway"
    candidates = [(good, None), (bad, None)]

    baseline = score_instruction_against_corpus(instruction, candidates)
    assert baseline.best_language == good

    layer = FeedbackLayer(env_id="test-env")
    layer.record(instruction, good, correct=True, source="user")
    layer.prepare(TFIDFEncoder().fit([instruction, good, bad]))

    boosted = score_instruction_against_corpus(
        instruction,
        candidates,
        feedback_layer=layer,
    )
    good_cand = next(c for c in boosted.candidates if c.language == good)
    assert good_cand.adjusted_score >= good_cand.base_score
    assert boosted.best_language == good


def test_feedback_layer_penalizes_incorrect_pair() -> None:
    instruction = "pick up the golden key"
    wrong = "empty corridor with no items"
    alt = "room containing a silver key on the table"
    candidates = [(wrong, None), (alt, None)]

    layer = FeedbackLayer(env_id="test-env")
    layer.record(instruction, wrong, correct=False, source="user")

    adjusted = score_instruction_against_corpus(
        instruction,
        candidates,
        feedback_layer=layer,
    )
    wrong_cand = next(c for c in adjusted.candidates if c.language == wrong)
    assert wrong_cand.adjusted_score <= wrong_cand.base_score


def test_feedback_persistence_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "match_feedback.json"
    record_match_feedback(
        "GridWorld-v0",
        "reach the goal tile",
        "agent on goal square",
        correct=True,
        source="user",
        path=str(path),
    )
    layer = get_feedback_layer("GridWorld-v0", path=str(path))
    assert len(layer.records) == 1
    assert layer.records[0].correct is True

    layer.record("fail case", "wrong tile", correct=False, source="llm")
    save_feedback_layer(layer)

    from rlip.instruction_matching import feedback as fb_mod

    fb_mod._FEEDBACK_STORE.clear()
    reloaded = get_feedback_layer("GridWorld-v0", path=str(path))
    assert len(reloaded.records) == 2
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["env_id"] == "GridWorld-v0"


def test_feedback_cache_reloads_when_disk_changes(tmp_path: Path) -> None:
    path = tmp_path / "match_feedback.json"
    record_match_feedback(
        "GridWorld-v0",
        "reach the goal tile",
        "agent on goal square",
        correct=True,
        source="user",
        path=str(path),
    )

    from rlip.instruction_matching import feedback as fb_mod

    cached = get_feedback_layer("GridWorld-v0", path=str(path))
    assert len(cached.records) == 1

    external = FeedbackLayer.from_dict(json.loads(path.read_text(encoding="utf-8")))
    external.record("another case", "other tile", correct=False, source="script")
    external._path = str(path)  # type: ignore[attr-defined]
    save_feedback_layer(external)

    reloaded = get_feedback_layer("GridWorld-v0", path=str(path))
    assert len(reloaded.records) == 2


def test_pair_direction_math() -> None:
    from rlip.instruction_matching.feedback import _pair_direction

    q = np.array([1.0, 0.0])
    s = np.array([0.0, 1.0])
    d = _pair_direction(q, s)
    assert d is not None
    assert np.allclose(np.linalg.norm(d), 1.0)
