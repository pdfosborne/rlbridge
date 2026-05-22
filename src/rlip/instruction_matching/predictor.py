"""
Supervised instruction-match predictor.

Learns from validated feedback whether an instruction matches a candidate state,
using text embeddings plus environment context.  Models are persisted per env_id
alongside the feedback JSON cache.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import BaseEncoder
from .feedback import FeedbackRecord, _cache_root, _safe_env_name
from .tfidf import TFIDFEncoder

INTERACT_DIM = 16
DIFF_DIM = 8
TAG_DIM = 8
DEFAULT_MIN_SAMPLES = 4
DEFAULT_LEARNING_RATE = 0.15
DEFAULT_L2 = 0.01
DEFAULT_EPOCHS = 80


@dataclass
class MatchContext:
    """Problem context passed into the supervised predictor."""

    env_id: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    namespace: str = ""

    def context_text(self) -> str:
        parts = [self.env_id]
        if self.description:
            parts.append(self.description)
        if self.tags:
            parts.append("tags: " + ", ".join(self.tags))
        if self.namespace:
            parts.append(f"domain: {self.namespace}")
        return ". ".join(parts)


@dataclass
class PredictorSample:
    instruction: str
    state_language: str
    label: bool
    timestamp: str = ""
    source: str = "user"

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "state_language": self.state_language,
            "label": self.label,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PredictorSample":
        return cls(
            instruction=d.get("instruction", ""),
            state_language=d.get("state_language", ""),
            label=bool(d.get("label", False)),
            timestamp=d.get("timestamp", ""),
            source=d.get("source", "user"),
        )

    @classmethod
    def from_feedback(cls, rec: FeedbackRecord) -> "PredictorSample":
        return cls(
            instruction=rec.instruction,
            state_language=rec.state_language,
            label=rec.correct,
            timestamp=rec.timestamp,
            source=rec.source,
        )


def default_predictor_path(env_id: str, *, cache_root: Path | None = None) -> Path:
    root = cache_root if cache_root is not None else _cache_root()
    return root / "environments" / _safe_env_name(env_id) / "match_predictor.pkl"


def resolve_match_context(env_id: str) -> MatchContext:
    """Resolve environment description/tags from catalog or registry."""
    description = ""
    tags: list[str] = []
    namespace = ""

    try:
        from ..public_registry.registry import public_registry

        entry = public_registry.get_entry(env_id)
        description = str(entry.get("description", "") or "")
        tags = list(entry.get("tags") or [])
        namespace = str(entry.get("namespace", "") or "")
    except Exception:  # noqa: BLE001
        pass

    if not description:
        try:
            from ..environments.registry import registry

            factory = registry.get(env_id)
            info = factory.env_info
            description = info.description or ""
            tags = list(info.tags or [])
            namespace = info.namespace or ""
        except Exception:  # noqa: BLE001
            pass

    return MatchContext(
        env_id=env_id,
        description=description,
        tags=tags,
        namespace=namespace,
    )


def _tag_features(tags: list[str]) -> np.ndarray:
    vec = np.zeros(TAG_DIM, dtype=np.float64)
    for tag in tags:
        idx = hash(tag.lower()) % TAG_DIM
        vec[idx] = 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def build_feature_vector(
    instruction_vec: np.ndarray,
    state_vec: np.ndarray,
    context_vec: np.ndarray,
    *,
    tags: list[str] | None = None,
) -> np.ndarray:
    """Fixed-size feature vector from embeddings and context tags."""
    q = np.asarray(instruction_vec, dtype=np.float64).ravel()
    s = np.asarray(state_vec, dtype=np.float64).ravel()
    c = np.asarray(context_vec, dtype=np.float64).ravel()
    dim = min(len(q), len(s))
    if dim == 0:
        base = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q = q[:dim]
        s = s[:dim]
        c = c[: min(dim, len(c))]
        if len(c) < dim:
            c = np.pad(c, (0, dim - len(c)))
        interact = (q * s)[:INTERACT_DIM]
        diff = (q - s)[:DIFF_DIM]
        if len(interact) < INTERACT_DIM:
            interact = np.pad(interact, (0, INTERACT_DIM - len(interact)))
        if len(diff) < DIFF_DIM:
            diff = np.pad(diff, (0, DIFF_DIM - len(diff)))
        base = np.concatenate(
            [
                np.array(
                    [float(np.dot(q, s)), float(np.dot(q, c)), float(np.dot(s, c))],
                    dtype=np.float64,
                ),
                interact,
                diff,
            ]
        )
    tag_vec = _tag_features(tags or [])
    return np.concatenate([base, tag_vec])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


@dataclass
class MatchPredictor:
    """
    Lightweight logistic-regression predictor trained from feedback samples.

    Features combine instruction/state/context embeddings with interaction
    terms and hashed tag indicators.
    """

    env_id: str
    context: MatchContext = field(default_factory=lambda: MatchContext(env_id=""))
    samples: list[PredictorSample] = field(default_factory=list)
    weights: np.ndarray | None = field(default=None, repr=False)
    bias: float = 0.0
    min_samples: int = DEFAULT_MIN_SAMPLES
    learning_rate: float = DEFAULT_LEARNING_RATE
    l2: float = DEFAULT_L2
    epochs: int = DEFAULT_EPOCHS
    _encoder: BaseEncoder | None = field(default=None, repr=False)
    _feature_dim: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not self.context.env_id:
            self.context.env_id = self.env_id

    def is_ready(self) -> bool:
        if len(self.samples) < self.min_samples:
            return False
        labels = [s.label for s in self.samples]
        return any(labels) and not all(labels)

    def blend_weight(self) -> float:
        if not self.is_ready():
            return 0.0
        n = len(self.samples)
        return min(0.6, n / (n + 8.0))

    def _fit_encoder(self, encoder: BaseEncoder | None = None) -> BaseEncoder:
        enc = encoder if encoder is not None else self._encoder
        if enc is None:
            enc = TFIDFEncoder()
        corpus = [self.context.context_text()]
        for sample in self.samples:
            corpus.extend([sample.instruction, sample.state_language])
        enc.fit(corpus)
        self._encoder = enc
        return enc

    def _encode_triplet(
        self,
        instruction: str,
        state_language: str,
        encoder: BaseEncoder,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ctx_text = self.context.context_text()
        return (
            encoder.encode(instruction),
            encoder.encode(state_language),
            encoder.encode(ctx_text),
        )

    def _features_for_pair(
        self,
        instruction: str,
        state_language: str,
        encoder: BaseEncoder,
    ) -> np.ndarray:
        q, s, c = self._encode_triplet(instruction, state_language, encoder)
        return build_feature_vector(q, s, c, tags=self.context.tags)

    def _ensure_weights(self, feature_dim: int) -> None:
        if self.weights is None or len(self.weights) != feature_dim:
            self.weights = np.zeros(feature_dim, dtype=np.float64)
            self.bias = 0.0
        self._feature_dim = feature_dim

    def _train(self, encoder: BaseEncoder | None = None) -> None:
        if len(self.samples) < 2:
            self.weights = None
            self.bias = 0.0
            return

        enc = self._fit_encoder(encoder)
        xs: list[np.ndarray] = []
        ys: list[float] = []
        for sample in self.samples:
            xs.append(self._features_for_pair(sample.instruction, sample.state_language, enc))
            ys.append(1.0 if sample.label else 0.0)

        x_mat = np.vstack(xs)
        y_vec = np.asarray(ys, dtype=np.float64)
        self._ensure_weights(x_mat.shape[1])

        w = self.weights.copy()
        b = self.bias
        lr = self.learning_rate
        l2 = self.l2
        for _ in range(self.epochs):
            logits = x_mat @ w + b
            preds = _sigmoid(logits)
            error = preds - y_vec
            grad_w = (x_mat.T @ error) / len(y_vec) + l2 * w
            grad_b = float(np.mean(error))
            w -= lr * grad_w
            b -= lr * grad_b

        self.weights = w
        self.bias = b

    def add_sample(
        self,
        instruction: str,
        state_language: str,
        *,
        label: bool,
        source: str = "user",
        encoder: BaseEncoder | None = None,
        retrain: bool = True,
    ) -> PredictorSample:
        sample = PredictorSample(
            instruction=instruction,
            state_language=state_language,
            label=label,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            source=source,
        )
        self.samples.append(sample)
        if retrain:
            self._train(encoder)
        return sample

    def add_feedback(
        self,
        rec: FeedbackRecord,
        *,
        encoder: BaseEncoder | None = None,
        retrain: bool = True,
    ) -> PredictorSample:
        return self.add_sample(
            rec.instruction,
            rec.state_language,
            label=rec.correct,
            source=rec.source,
            encoder=encoder,
            retrain=retrain,
        )

    def predict_proba(
        self,
        instruction: str,
        state_language: str,
        *,
        encoder: BaseEncoder | None = None,
    ) -> float:
        if not self.is_ready() or self.weights is None:
            return 0.5
        enc = self._fit_encoder(encoder)
        x = self._features_for_pair(instruction, state_language, enc)
        if len(x) != len(self.weights):
            return 0.5
        logit = float(np.dot(self.weights, x) + self.bias)
        return float(_sigmoid(np.array([logit]))[0])

    @staticmethod
    def proba_to_score(proba: float) -> float:
        """Map match probability to a cosine-like score in [0, 1]."""
        return float(np.clip(proba, 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "context": {
                "env_id": self.context.env_id,
                "description": self.context.description,
                "tags": self.context.tags,
                "namespace": self.context.namespace,
            },
            "samples": [s.to_dict() for s in self.samples],
            "weights": self.weights.tolist() if self.weights is not None else None,
            "bias": self.bias,
            "min_samples": self.min_samples,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "epochs": self.epochs,
            "feature_dim": self._feature_dim,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MatchPredictor":
        ctx_raw = d.get("context") or {}
        context = MatchContext(
            env_id=ctx_raw.get("env_id", d.get("env_id", "")),
            description=ctx_raw.get("description", ""),
            tags=list(ctx_raw.get("tags") or []),
            namespace=ctx_raw.get("namespace", ""),
        )
        weights_raw = d.get("weights")
        weights = np.asarray(weights_raw, dtype=np.float64) if weights_raw else None
        predictor = cls(
            env_id=d.get("env_id", context.env_id),
            context=context,
            samples=[PredictorSample.from_dict(s) for s in (d.get("samples") or [])],
            weights=weights,
            bias=float(d.get("bias", 0.0)),
            min_samples=int(d.get("min_samples", DEFAULT_MIN_SAMPLES)),
            learning_rate=float(d.get("learning_rate", DEFAULT_LEARNING_RATE)),
            l2=float(d.get("l2", DEFAULT_L2)),
            epochs=int(d.get("epochs", DEFAULT_EPOCHS)),
        )
        predictor._feature_dim = int(d.get("feature_dim", len(weights) if weights is not None else 0))
        return predictor


_PREDICTOR_STORE: dict[str, MatchPredictor] = {}


def _load_predictor(env_id: str, store_path: str) -> MatchPredictor:
    path_obj = Path(store_path)
    predictor = MatchPredictor(env_id=env_id, context=resolve_match_context(env_id))
    disk_mtime: float | None = None
    if path_obj.exists():
        try:
            raw = pickle.loads(path_obj.read_bytes())
            if isinstance(raw, dict):
                predictor = MatchPredictor.from_dict(raw)
            predictor.env_id = env_id
            if not predictor.context.description:
                predictor.context = resolve_match_context(env_id)
            disk_mtime = path_obj.stat().st_mtime
        except Exception:  # noqa: BLE001
            predictor = MatchPredictor(env_id=env_id, context=resolve_match_context(env_id))

    predictor._path = store_path  # type: ignore[attr-defined]
    predictor._disk_mtime = disk_mtime  # type: ignore[attr-defined]
    return predictor


def get_match_predictor(
    env_id: str,
    *,
    path: str | None = None,
    cache_root: Path | None = None,
) -> MatchPredictor:
    store_path = path or str(default_predictor_path(env_id, cache_root=cache_root))
    path_obj = Path(store_path)

    if env_id in _PREDICTOR_STORE:
        predictor = _PREDICTOR_STORE[env_id]
        cached_path = getattr(predictor, "_path", None)
        if cached_path != store_path:
            predictor = _load_predictor(env_id, store_path)
            _PREDICTOR_STORE[env_id] = predictor
            return predictor
        if path_obj.exists():
            disk_mtime = path_obj.stat().st_mtime
            cached_mtime = getattr(predictor, "_disk_mtime", None)
            if cached_mtime is None or disk_mtime > cached_mtime:
                predictor = _load_predictor(env_id, store_path)
                _PREDICTOR_STORE[env_id] = predictor
        return predictor

    predictor = _load_predictor(env_id, store_path)
    _PREDICTOR_STORE[env_id] = predictor
    return predictor


def save_match_predictor(predictor: MatchPredictor) -> None:
    path = getattr(predictor, "_path", None)
    if not path:
        path = str(default_predictor_path(predictor.env_id))
        predictor._path = path  # type: ignore[attr-defined]
    store = Path(path)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_bytes(pickle.dumps(predictor.to_dict(), protocol=pickle.HIGHEST_PROTOCOL))
    predictor._disk_mtime = store.stat().st_mtime  # type: ignore[attr-defined]


def update_predictor_from_feedback(
    env_id: str,
    rec: FeedbackRecord,
    *,
    path: str | None = None,
    encoder: BaseEncoder | None = None,
) -> MatchPredictor:
    predictor = get_match_predictor(env_id, path=path)
    if not predictor.context.description:
        predictor.context = resolve_match_context(env_id)
    predictor.add_feedback(rec, encoder=encoder)
    save_match_predictor(predictor)
    return predictor


def export_predictor_json(predictor: MatchPredictor) -> str:
    """Return a JSON snapshot of predictor metadata (weights included)."""
    return json.dumps(predictor.to_dict(), indent=2)


__all__ = [
    "MatchContext",
    "MatchPredictor",
    "PredictorSample",
    "build_feature_vector",
    "default_predictor_path",
    "resolve_match_context",
    "get_match_predictor",
    "save_match_predictor",
    "update_predictor_from_feedback",
    "export_predictor_json",
]
