"""
Feedback layer for instruction matching.

Validated (instruction, state) pairs adjust future TF-IDF cosine scores via
aggregate *feedback layer vectors* and per-entry pair similarity.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import BaseEncoder


def _cache_root() -> Path:
    env_override = os.environ.get("rlbridge_CACHE_ROOT") or os.environ.get("rlbridge_OUTPUT_ROOT")
    if env_override:
        root = Path(env_override) / ".rlbridge"
    else:
        root = Path.home() / ".rlbridge"
    return root


def _safe_env_name(env_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", env_id).strip("._-")
    return cleaned or "environment"


def default_feedback_path(env_id: str, *, cache_root: Path | None = None) -> Path:
    """Return the default JSON path for per-environment match feedback."""
    root = cache_root if cache_root is not None else _cache_root()
    return root / "environments" / _safe_env_name(env_id) / "match_feedback.json"


@dataclass
class FeedbackRecord:
    """One validated instruction-to-state match."""

    instruction: str
    state_language: str
    correct: bool
    timestamp: str = ""
    source: str = "user"
    """Who validated: ``"user"``, ``"llm"``, or ``"script"``."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "state_language": self.state_language,
            "correct": self.correct,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeedbackRecord":
        return cls(
            instruction=d.get("instruction", ""),
            state_language=d.get("state_language", ""),
            correct=bool(d.get("correct", False)),
            timestamp=d.get("timestamp", ""),
            source=d.get("source", "user"),
        )


@dataclass
class FeedbackLayer:
    """
    Adjusts TF-IDF cosine similarity using validated match feedback.

    Mathematics
    -----------
    For each candidate pair with instruction vector *q* and state vector *s*:

    1. **Baseline** - standard TF-IDF cosine similarity::

           base = dot(q, s)

    2. **Pair direction** - unit vector pointing from instruction to state::

           d = normalize(s - q)

    3. **Layer vectors** - running sums of *d* over validated pairs::

           P = Σ d_i   (correct pairs)
           N = Σ d_j   (incorrect pairs)

       Layer adjustment (when norms are non-zero)::

           layer_adj = boost_w * dot(d, P/||P||) - penalty_w * dot(d, N/||N||)

    4. **Entry adjustment** - for each stored feedback record, compute pair
       overlap with the current candidate::

           pair_sim = 0.5 * (dot(q, q') + dot(s, s'))

       Sum boosts for correct records and penalties for incorrect ones::

           entry_adj = Σ (± entry_w * pair_sim)

    5. **Final score**::

           adjusted = clip(base + layer_adj + entry_adj, -1, 1)
    """

    env_id: str
    boost_weight: float = 0.12
    penalty_weight: float = 0.18
    entry_weight: float = 0.08
    records: list[FeedbackRecord] = field(default_factory=list)
    _positive_layer: np.ndarray | None = field(default=None, repr=False)
    _negative_layer: np.ndarray | None = field(default=None, repr=False)
    _prepared: bool = field(default=False, repr=False)

    def record(
        self,
        instruction: str,
        state_language: str,
        *,
        correct: bool,
        source: str = "user",
    ) -> FeedbackRecord:
        """Append a validation result and invalidate cached layer vectors."""
        rec = FeedbackRecord(
            instruction=instruction,
            state_language=state_language,
            correct=correct,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            source=source,
        )
        self.records.append(rec)
        self._prepared = False
        self._positive_layer = None
        self._negative_layer = None
        return rec

    def prepare(self, encoder: BaseEncoder) -> None:
        """Encode stored records and rebuild aggregate layer vectors."""
        dim = 0
        for rec in self.records:
            try:
                q = encoder.encode(rec.instruction)
                s = encoder.encode(rec.state_language)
            except RuntimeError:
                continue
            dim = max(dim, len(q), len(s))
        if dim == 0:
            self._positive_layer = None
            self._negative_layer = None
            self._prepared = True
            return

        positive = np.zeros(dim, dtype=np.float64)
        negative = np.zeros(dim, dtype=np.float64)
        for rec in self.records:
            try:
                q = encoder.encode(rec.instruction)
                s = encoder.encode(rec.state_language)
            except RuntimeError:
                continue
            delta = _pair_direction(q, s)
            if delta is None:
                continue
            if rec.correct:
                positive += delta
            else:
                negative += delta

        self._positive_layer = positive if np.linalg.norm(positive) > 0 else None
        self._negative_layer = negative if np.linalg.norm(negative) > 0 else None
        self._prepared = True

    def adjust_score(
        self,
        instruction_vec: np.ndarray,
        state_vec: np.ndarray,
        base_score: float,
        *,
        encoder: BaseEncoder | None = None,
    ) -> float:
        """Return feedback-adjusted similarity for one candidate pair."""
        if encoder is not None and not self._prepared:
            self.prepare(encoder)

        adjustment = 0.0
        pair_dir = _pair_direction(instruction_vec, state_vec)
        if pair_dir is not None:
            if self._positive_layer is not None:
                norm = float(np.linalg.norm(self._positive_layer))
                if norm > 0:
                    adjustment += self.boost_weight * float(
                        np.dot(pair_dir, self._positive_layer / norm)
                    )
            if self._negative_layer is not None:
                norm = float(np.linalg.norm(self._negative_layer))
                if norm > 0:
                    adjustment -= self.penalty_weight * float(
                        np.dot(pair_dir, self._negative_layer / norm)
                    )

        if encoder is not None:
            for rec in self.records:
                try:
                    q_ref = encoder.encode(rec.instruction)
                    s_ref = encoder.encode(rec.state_language)
                except RuntimeError:
                    continue
                pair_sim = _pair_overlap(instruction_vec, state_vec, q_ref, s_ref)
                if rec.correct:
                    adjustment += self.entry_weight * pair_sim
                else:
                    adjustment -= self.entry_weight * pair_sim

        return float(np.clip(base_score + adjustment, -1.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "boost_weight": self.boost_weight,
            "penalty_weight": self.penalty_weight,
            "entry_weight": self.entry_weight,
            "records": [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeedbackLayer":
        return cls(
            env_id=d.get("env_id", ""),
            boost_weight=float(d.get("boost_weight", 0.12)),
            penalty_weight=float(d.get("penalty_weight", 0.18)),
            entry_weight=float(d.get("entry_weight", 0.08)),
            records=[FeedbackRecord.from_dict(r) for r in (d.get("records") or [])],
        )


_FEEDBACK_STORE: dict[str, FeedbackLayer] = {}


def _load_feedback_layer(env_id: str, store_path: str) -> FeedbackLayer:
    """Load a feedback layer from *store_path*, or return an empty layer."""
    layer = FeedbackLayer(env_id=env_id)
    path_obj = Path(store_path)
    disk_mtime: float | None = None
    if path_obj.exists():
        try:
            raw = json.loads(path_obj.read_text(encoding="utf-8"))
            layer = FeedbackLayer.from_dict(raw)
            layer.env_id = env_id
            disk_mtime = path_obj.stat().st_mtime
        except Exception:  # noqa: BLE001
            pass

    layer._path = store_path  # type: ignore[attr-defined]
    layer._disk_mtime = disk_mtime  # type: ignore[attr-defined]
    return layer


def get_feedback_layer(
    env_id: str,
    *,
    path: str | None = None,
    cache_root: Path | None = None,
) -> FeedbackLayer:
    """Return the feedback layer for *env_id*, loading from disk when needed."""
    store_path = path or str(default_feedback_path(env_id, cache_root=cache_root))
    path_obj = Path(store_path)

    if env_id in _FEEDBACK_STORE:
        layer = _FEEDBACK_STORE[env_id]
        cached_path = getattr(layer, "_path", None)
        if cached_path != store_path:
            layer = _load_feedback_layer(env_id, store_path)
            _FEEDBACK_STORE[env_id] = layer
            return layer
        if path_obj.exists():
            disk_mtime = path_obj.stat().st_mtime
            cached_mtime = getattr(layer, "_disk_mtime", None)
            if cached_mtime is None or disk_mtime > cached_mtime:
                layer = _load_feedback_layer(env_id, store_path)
                _FEEDBACK_STORE[env_id] = layer
        return layer

    layer = _load_feedback_layer(env_id, store_path)
    _FEEDBACK_STORE[env_id] = layer
    return layer


def save_feedback_layer(layer: FeedbackLayer) -> None:
    """Persist *layer* to its configured path."""
    path = getattr(layer, "_path", None)
    if not path:
        path = str(default_feedback_path(layer.env_id))
        layer._path = path  # type: ignore[attr-defined]
    store = Path(path)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(layer.to_dict(), indent=2), encoding="utf-8")
    layer._disk_mtime = store.stat().st_mtime  # type: ignore[attr-defined]


def record_match_feedback(
    env_id: str,
    instruction: str,
    state_language: str,
    *,
    correct: bool,
    source: str = "user",
    path: str | None = None,
    update_predictor: bool = True,
    encoder: BaseEncoder | None = None,
) -> FeedbackRecord:
    """Record validation feedback and persist it."""
    layer = get_feedback_layer(env_id, path=path)
    rec = layer.record(instruction, state_language, correct=correct, source=source)
    save_feedback_layer(layer)
    if update_predictor:
        from .predictor import update_predictor_from_feedback

        predictor_path = None
        if path:
            predictor_path = str(Path(path).with_name("match_predictor.pkl"))
        update_predictor_from_feedback(env_id, rec, path=predictor_path, encoder=encoder)
    return rec


def _pair_direction(q: np.ndarray, s: np.ndarray) -> np.ndarray | None:
    delta = s - q
    norm = float(np.linalg.norm(delta))
    if norm <= 0.0:
        return None
    return delta / norm


def _pair_overlap(
    q: np.ndarray,
    s: np.ndarray,
    q_ref: np.ndarray,
    s_ref: np.ndarray,
) -> float:
    if len(q) == 0 or len(s) == 0:
        return 0.0
    q_sim = float(np.dot(q, q_ref)) if len(q) == len(q_ref) else 0.0
    s_sim = float(np.dot(s, s_ref)) if len(s) == len(s_ref) else 0.0
    return 0.5 * (q_sim + s_sim)


__all__ = [
    "FeedbackRecord",
    "FeedbackLayer",
    "default_feedback_path",
    "get_feedback_layer",
    "save_feedback_layer",
    "record_match_feedback",
]
