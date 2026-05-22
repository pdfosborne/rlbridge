"""
Instruction matching — two-stage TF-IDF filter + semantic re-rank.

Stage 1 uses TF-IDF cosine similarity to shortlist top candidates from the
full corpus.  Stage 2 re-scores that subset with a dense encoder (sentence
transformers by default).  An optional :class:`FeedbackLayer` adjusts the
refined scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .base import BaseEncoder
from .feedback import FeedbackLayer
from .predictor import MatchContext, MatchPredictor
from .tfidf import TFIDFEncoder, TextEncoder

DEFAULT_REFINE_TOP_K = 20


@dataclass
class ScoredCandidate:
    """One observed state ranked against an instruction."""

    language: str
    observation: Any
    base_score: float
    """Encoder similarity before feedback adjustment."""

    adjusted_score: float
    """Similarity after feedback and supervised blending."""

    supervised_score: float | None = None
    """Supervised match probability mapped to [0, 1], or *None* when inactive."""


@dataclass
class CorpusMatchResult:
    """Outcome of matching an instruction against a pre-built language corpus."""

    instruction: str
    best_language: str
    best_observation: Any
    similarity_score: float
    """Best *adjusted* score (used for ranking and sub-goal selection)."""

    base_similarity_score: float
    """Best raw encoder score before feedback adjustment."""

    matched_states: list[tuple[str, Any, float]] = field(default_factory=list)
    """``(language, observation, adjusted_score)`` within *similarity_band*."""

    all_scores: list[tuple[str, float]] = field(default_factory=list)
    """All ``(language, adjusted_score)`` pairs, descending."""

    candidates: list[ScoredCandidate] = field(default_factory=list, repr=False)
    """Full per-candidate detail, descending by *adjusted_score*."""


def _is_tfidf_encoder(encoder: BaseEncoder) -> bool:
    return isinstance(encoder, (TFIDFEncoder, TextEncoder))


def _default_refine_encoder() -> BaseEncoder:
    from . import get_encoder

    return get_encoder("sentence")


def _blend_supervised_score(
    feedback_score: float,
    instruction: str,
    state_language: str,
    *,
    predictor: MatchPredictor | None,
    match_context: MatchContext | None,
) -> tuple[float, float | None]:
    if predictor is None or not predictor.is_ready():
        return feedback_score, None

    ctx = match_context or predictor.context
    if ctx.env_id and not predictor.context.description:
        predictor.context = ctx

    proba = predictor.predict_proba(instruction, state_language)
    supervised = MatchPredictor.proba_to_score(proba)
    weight = predictor.blend_weight()
    blended = (1.0 - weight) * feedback_score + weight * supervised
    return float(np.clip(blended, -1.0, 1.0)), supervised


def score_instruction_against_corpus(
    instruction: str,
    candidates: list[tuple[str, Any]],
    *,
    encoder: Optional[BaseEncoder] = None,
    feedback_layer: Optional[FeedbackLayer] = None,
    predictor: MatchPredictor | None = None,
    match_context: MatchContext | None = None,
    similarity_band: float = 0.05,
    refine_top_k: int = DEFAULT_REFINE_TOP_K,
) -> CorpusMatchResult:
    """
    Match *instruction* to *candidates* using two-stage scoring.

    Stage 1 ranks every candidate with TF-IDF cosine similarity and keeps the
    top *refine_top_k*.  Stage 2 re-scores that subset with a dense semantic
    encoder (sentence-transformers by default).  Remaining candidates keep
    their TF-IDF scores and are ranked below the refined set.

    When *encoder* is a :class:`TFIDFEncoder` (or legacy :class:`TextEncoder`),
    only TF-IDF scoring is used — useful for fast tests and backward
    compatibility.

    Parameters
    ----------
    instruction:
        Natural-language goal text.
    candidates:
        ``(language_description, raw_observation)`` pairs observed in the
        environment.  Duplicate language strings should be deduplicated by
        the caller.
    encoder:
        Stage-2 refine encoder.  Defaults to sentence-transformers
        (``all-MiniLM-L6-v2``).  Pass :class:`TFIDFEncoder` to disable
        semantic re-ranking.
    feedback_layer:
        Optional validated-feedback layer that boosts or penalises scores.
        Applied on top of refine-encoder vectors for shortlisted candidates.
    predictor:
        Optional supervised model trained from validation feedback.  When
        enough labelled samples exist, its probability is blended into the
        final ranking score.
    match_context:
        Environment context (description, tags) for the supervised predictor.
    similarity_band:
        Include every candidate whose adjusted score is within this margin of
        the best score.
    refine_top_k:
        Number of TF-IDF top candidates to re-score with the refine encoder.

    Returns
    -------
    CorpusMatchResult
    """
    if not candidates:
        raise ValueError("candidates must contain at least one (language, observation) pair")

    unique_langs = [lang for lang, _ in candidates]
    unique_obs = [obs for _, obs in candidates]

    filter_encoder = TFIDFEncoder()
    refine_encoder: BaseEncoder = encoder if encoder is not None else _default_refine_encoder()
    two_stage = not _is_tfidf_encoder(refine_encoder)

    corpus = [instruction] + unique_langs
    filter_encoder.fit(corpus)
    instruction_filter_vec = filter_encoder.encode(instruction)

    tfidf_ranked: list[tuple[str, Any, float]] = []
    for lang, obs in zip(unique_langs, unique_obs):
        obs_vec = filter_encoder.encode(lang)
        tfidf_ranked.append(
            (lang, obs, filter_encoder.cosine_similarity(instruction_filter_vec, obs_vec))
        )
    tfidf_ranked.sort(key=lambda row: row[2], reverse=True)

    refine_base: dict[str, float] = {}
    instruction_refine_vec: np.ndarray | None = None

    if two_stage:
        k = max(1, min(refine_top_k, len(tfidf_ranked)))
        shortlist = tfidf_ranked[:k]
        refine_corpus = [instruction] + [lang for lang, _, _ in shortlist]
        refine_encoder.fit(refine_corpus)
        instruction_refine_vec = refine_encoder.encode(instruction)

        if feedback_layer is not None:
            feedback_layer.prepare(refine_encoder)

        for lang, _, _ in shortlist:
            obs_vec = refine_encoder.encode(lang)
            refine_base[lang] = refine_encoder.cosine_similarity(
                instruction_refine_vec, obs_vec
            )
    else:
        refine_encoder = filter_encoder
        instruction_refine_vec = instruction_filter_vec
        if feedback_layer is not None:
            feedback_layer.prepare(refine_encoder)

    scored: list[ScoredCandidate] = []
    for lang, obs, tfidf_score in tfidf_ranked:
        if lang in refine_base:
            base = refine_base[lang]
            score_instruction_vec = instruction_refine_vec  # type: ignore[arg-type]
            score_state_vec = refine_encoder.encode(lang)
            adjusted = (
                feedback_layer.adjust_score(
                    score_instruction_vec,
                    score_state_vec,
                    base,
                    encoder=refine_encoder,
                )
                if feedback_layer is not None
                else base
            )
        else:
            base = tfidf_score
            adjusted = base

        adjusted, supervised = _blend_supervised_score(
            adjusted,
            instruction,
            lang,
            predictor=predictor,
            match_context=match_context,
        )

        scored.append(
            ScoredCandidate(
                language=lang,
                observation=obs,
                base_score=base,
                adjusted_score=adjusted,
                supervised_score=supervised,
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
    "DEFAULT_REFINE_TOP_K",
    "ScoredCandidate",
    "CorpusMatchResult",
    "score_instruction_against_corpus",
]
