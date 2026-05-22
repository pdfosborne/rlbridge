"""
Instruction matching — TF-IDF cosine baseline with optional feedback layer.

This module owns the text-similarity scoring step.  Environment exploration
and language translation remain in :mod:`rlip.instruction_following`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .base import BaseEncoder
from .feedback import FeedbackLayer
from .tfidf import TFIDFEncoder


@dataclass
class ScoredCandidate:
    """One observed state ranked against an instruction."""

    language: str
    observation: Any
    base_score: float
    """TF-IDF cosine similarity before feedback adjustment."""

    adjusted_score: float
    """Similarity after :class:`FeedbackLayer` adjustment (equals *base_score*
    when no feedback layer is supplied)."""


@dataclass
class CorpusMatchResult:
    """Outcome of matching an instruction against a pre-built language corpus."""

    instruction: str
    best_language: str
    best_observation: Any
    similarity_score: float
    """Best *adjusted* score (used for ranking and sub-goal selection)."""

    base_similarity_score: float
    """Best raw TF-IDF cosine score."""

    matched_states: list[tuple[str, Any, float]] = field(default_factory=list)
    """``(language, observation, adjusted_score)`` within *similarity_band*."""

    all_scores: list[tuple[str, float]] = field(default_factory=list)
    """All ``(language, adjusted_score)`` pairs, descending."""

    candidates: list[ScoredCandidate] = field(default_factory=list, repr=False)
    """Full per-candidate detail, descending by *adjusted_score*."""


def score_instruction_against_corpus(
    instruction: str,
    candidates: list[tuple[str, Any]],
    *,
    encoder: Optional[BaseEncoder] = None,
    feedback_layer: Optional[FeedbackLayer] = None,
    similarity_band: float = 0.05,
) -> CorpusMatchResult:
    """
    Match *instruction* to *candidates* using TF-IDF cosine similarity.

    Parameters
    ----------
    instruction:
        Natural-language goal text.
    candidates:
        ``(language_description, raw_observation)`` pairs observed in the
        environment.  Duplicate language strings should be deduplicated by
        the caller.
    encoder:
        Text encoder.  Defaults to :class:`TFIDFEncoder`.
    feedback_layer:
        Optional validated-feedback layer that boosts or penalises scores.
    similarity_band:
        Include every candidate whose adjusted score is within this margin of
        the best score.

    Returns
    -------
    CorpusMatchResult
    """
    if not candidates:
        raise ValueError("candidates must contain at least one (language, observation) pair")

    unique_langs = [lang for lang, _ in candidates]
    unique_obs = [obs for _, obs in candidates]

    _encoder: BaseEncoder = encoder if encoder is not None else TFIDFEncoder()
    corpus = [instruction] + unique_langs
    _encoder.fit(corpus)
    instruction_vec = _encoder.encode(instruction)

    if feedback_layer is not None:
        feedback_layer.prepare(_encoder)

    scored: list[ScoredCandidate] = []
    for lang, obs in zip(unique_langs, unique_obs):
        obs_vec = _encoder.encode(lang)
        base = _encoder.cosine_similarity(instruction_vec, obs_vec)
        adjusted = (
            feedback_layer.adjust_score(instruction_vec, obs_vec, base, encoder=_encoder)
            if feedback_layer is not None
            else base
        )
        scored.append(
            ScoredCandidate(
                language=lang,
                observation=obs,
                base_score=base,
                adjusted_score=adjusted,
            )
        )

    scored.sort(key=lambda c: c.adjusted_score, reverse=True)
    best = scored[0]
    all_scores = [(c.language, c.adjusted_score) for c in scored]

    cutoff = best.adjusted_score - similarity_band
    matched_states = [
        (c.language, c.observation, c.adjusted_score)
        for c in scored
        if c.adjusted_score >= cutoff
    ]

    return CorpusMatchResult(
        instruction=instruction,
        best_language=best.language,
        best_observation=best.observation,
        similarity_score=best.adjusted_score,
        base_similarity_score=best.base_score,
        matched_states=matched_states,
        all_scores=all_scores,
        candidates=scored,
    )


__all__ = [
    "ScoredCandidate",
    "CorpusMatchResult",
    "score_instruction_against_corpus",
]
