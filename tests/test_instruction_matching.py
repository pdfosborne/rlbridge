"""Tests for two-stage instruction matching."""

from __future__ import annotations

import numpy as np
import pytest

from rlip.instruction_matching.feedback import FeedbackLayer
from rlip.instruction_matching.matcher import (
    DEFAULT_REFINE_TOP_K,
    score_instruction_against_corpus,
)
from rlip.instruction_matching.tfidf import TFIDFEncoder


class _KeywordRefineEncoder:
    """Deterministic semantic encoder for tests (no model download)."""

    _TOPICS = {
        "beach": np.array([1.0, 0.0, 0.0]),
        "shore": np.array([0.95, 0.05, 0.0]),
        "harbor": np.array([0.0, 1.0, 0.0]),
        "dock": np.array([0.0, 0.95, 0.05]),
        "water": np.array([0.0, 0.0, 1.0]),
        "ocean": np.array([0.0, 0.05, 0.95]),
    }

    def fit(self, corpus: list[str]) -> "_KeywordRefineEncoder":
        self._corpus = corpus
        return self

    def encode(self, text: str) -> np.ndarray:
        lowered = text.lower()
        for keyword, vec in self._TOPICS.items():
            if keyword in lowered:
                return vec / np.linalg.norm(vec)
        return np.array([0.33, 0.33, 0.34])

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


def _pairs(*langs: str) -> list[tuple[str, None]]:
    return [(lang, None) for lang in langs]


def test_two_stage_prefers_semantic_match_over_tfidf():
    instruction = "sail towards the beach"
    candidates = _pairs(
        "agent is in open water far from land",
        "agent is near the sandy shore",
        "agent is close to the harbor dock",
    )

    result = score_instruction_against_corpus(
        instruction,
        candidates,
        encoder=_KeywordRefineEncoder(),
        refine_top_k=2,
    )

    assert result.best_language == "agent is near the sandy shore"
    assert result.candidates[0].base_score > result.candidates[-1].base_score


def test_tfidf_only_mode_uses_single_encoder():
    instruction = "sail towards the beach"
    candidates = _pairs("near the shore", "in open water")

    result = score_instruction_against_corpus(
        instruction,
        candidates,
        encoder=TFIDFEncoder(),
    )

    assert result.best_language in {lang for lang, _ in candidates}
    assert len(result.candidates) == 2


def test_refine_top_k_limits_semantic_rescore():
    instruction = "go to the beach"
    candidates = _pairs(
        "state on the sandy shore",
        "state in the harbor area",
        "state in deep ocean water",
        "state at the dock",
    )

    full = score_instruction_against_corpus(
        instruction,
        candidates,
        encoder=_KeywordRefineEncoder(),
        refine_top_k=len(candidates),
    )
    limited = score_instruction_against_corpus(
        instruction,
        candidates,
        encoder=_KeywordRefineEncoder(),
        refine_top_k=1,
    )

    assert full.best_language == "state on the sandy shore"
    assert limited.best_language != "state on the sandy shore"


def test_feedback_adjusts_refined_scores():
    instruction = "sail towards the beach"
    candidates = _pairs("near the sandy shore", "in open ocean water")
    layer = FeedbackLayer(env_id="test")
    layer.record(instruction, "near the sandy shore", correct=True)

    without = score_instruction_against_corpus(
        instruction,
        candidates,
        encoder=_KeywordRefineEncoder(),
        feedback_layer=None,
    )
    with_fb = score_instruction_against_corpus(
        instruction,
        candidates,
        encoder=_KeywordRefineEncoder(),
        feedback_layer=layer,
    )

    shore_without = next(
        c for c in without.candidates if "shore" in c.language
    )
    shore_with = next(c for c in with_fb.candidates if "shore" in c.language)
    assert shore_with.adjusted_score >= shore_without.adjusted_score


def test_empty_candidates_raises():
    with pytest.raises(ValueError, match="at least one"):
        score_instruction_against_corpus("goal", [])


def test_default_refine_top_k_constant():
    assert DEFAULT_REFINE_TOP_K == 20
