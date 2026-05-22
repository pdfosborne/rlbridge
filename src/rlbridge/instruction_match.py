"""
rlbridge.instruction_match
======================
Programmatic corpus matching: rank natural-language candidate strings
against an instruction using two-stage scoring (TF-IDF + semantic re-rank),
with optional validated-feedback and supervised-predictor adjustments.

Unlike :func:`rlbridge.instruction_following.match_instruction`, this module
operates on a pre-built candidate list - no environment exploration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .instruction_matching import (
    BaseEncoder,
    TFIDFEncoder,
    get_feedback_layer,
    get_match_predictor,
    record_match_feedback,
    resolve_match_context,
    score_instruction_against_corpus,
)
from .instruction_matching.matcher import DEFAULT_REFINE_TOP_K, CorpusMatchResult

__all__ = [
    "DEFAULT_REFINE_TOP_K",
    "CorpusMatchResult",
    "load_candidates",
    "resolve_candidates",
    "match_candidates",
    "record_validation",
    "format_match_result",
]


def load_candidates(path: str | Path) -> list[str]:
    """Load candidate strings from a JSON file (list or ``{\"candidates\": [...]}``)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "candidates" in data:
        return [str(x) for x in data["candidates"]]
    raise ValueError(f"Unsupported candidates file format: {path}")


def resolve_candidates(
    candidates: list[str] | None = None,
    *,
    candidates_file: str | Path | None = None,
) -> list[str]:
    """Merge inline *candidates* with optional JSON file contents."""
    merged = list(candidates or [])
    if candidates_file:
        merged.extend(load_candidates(candidates_file))
    if not merged:
        raise ValueError("Provide candidates and/or candidates_file.")
    return merged


def match_candidates(
    instruction: str,
    candidates: list[str],
    *,
    env_id: str = "",
    encoder: BaseEncoder | None = None,
    similarity_band: float = 0.05,
    use_feedback: bool = True,
    use_predictor: bool = True,
    refine_top_k: int = DEFAULT_REFINE_TOP_K,
    tfidf_only: bool = False,
) -> CorpusMatchResult:
    """
    Rank *candidates* against *instruction* using two-stage matching.

    When *env_id* is set and *use_feedback* / *use_predictor* are enabled,
    loads persisted feedback layers and predictors for that environment.
    """
    if not candidates:
        raise ValueError("candidates must be a non-empty list of strings")

    env_id = env_id.strip()
    pairs: list[tuple[str, Any]] = [(lang, None) for lang in candidates]

    feedback = None
    if use_feedback and env_id:
        feedback = get_feedback_layer(env_id)

    predictor = None
    match_context = None
    if env_id and use_feedback and use_predictor:
        predictor = get_match_predictor(env_id)
        match_context = resolve_match_context(env_id)

    resolved_encoder = TFIDFEncoder() if tfidf_only else encoder

    return score_instruction_against_corpus(
        instruction,
        pairs,
        encoder=resolved_encoder,
        feedback_layer=feedback,
        predictor=predictor,
        match_context=match_context,
        similarity_band=similarity_band,
        refine_top_k=refine_top_k,
    )


def record_validation(
    env_id: str,
    instruction: str,
    state_language: str,
    *,
    correct: bool,
    source: str = "api",
) -> None:
    """Record validated match feedback for *env_id*."""
    if not env_id.strip():
        raise ValueError("env_id is required when recording validation feedback")
    record_match_feedback(
        env_id,
        instruction,
        state_language,
        correct=correct,
        source=source,
    )


def format_match_result(result: CorpusMatchResult, *, top_k: int = 5) -> str:
    """Return a human-readable summary of *result*."""
    k = max(1, min(top_k, len(result.candidates)))
    lines = [
        f"Instruction: {result.instruction!r}",
        f"Best match:  {result.best_language!r}",
        f"Adjusted:    {result.similarity_score:.4f}",
        f"Base score:  {result.base_similarity_score:.4f}",
        f"Sub-goals:   {len(result.matched_states)}",
        "",
        f"Top {k} candidates:",
    ]
    for i, cand in enumerate(result.candidates[:k], start=1):
        sup = (
            f"  sup={cand.supervised_score:.4f}"
            if cand.supervised_score is not None
            else ""
        )
        lines.append(
            f"  {i}. adj={cand.adjusted_score:.4f}  base={cand.base_score:.4f}{sup}  "
            f"{cand.language[:120]}"
        )
    return "\n".join(lines)
