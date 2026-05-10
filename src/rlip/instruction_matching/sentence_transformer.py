"""
SentenceEncoder — semantic instruction matcher via sentence-transformers.

Unlike the bag-of-words methods (TF-IDF, BM25), ``SentenceEncoder`` uses a
pre-trained deep neural network to produce dense *semantic* embeddings.  This
means it understands that "navigate to the shore" and "sail towards the beach"
express the same intent, even though they share no common tokens.

Requirements
------------
``sentence-transformers`` must be installed::

    pip install sentence-transformers

The default model (``all-MiniLM-L6-v2``) is small (80 MB), fast on CPU, and
performs well on general-purpose semantic similarity tasks.  Other options:

* ``all-mpnet-base-v2`` — higher accuracy, larger (420 MB)
* ``paraphrase-MiniLM-L3-v2`` — fastest, slightly lower accuracy

The model is downloaded automatically on first use and cached in the
Hugging Face hub cache (``~/.cache/huggingface/``).

Usage
-----
::

    enc = SentenceEncoder()             # or SentenceEncoder("all-mpnet-base-v2")
    enc.fit(corpus)                     # no-op; model is pre-trained
    q = enc.encode("sail to the beach")
    d = enc.encode("navigate towards shore")
    print(enc.cosine_similarity(q, d))  # high similarity (~0.7-0.9)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import BaseEncoder


class SentenceEncoder(BaseEncoder):
    """
    Semantic text encoder backed by a sentence-transformers model.

    Embeddings are L2-normalised by the underlying library, so
    :meth:`~base.BaseEncoder.cosine_similarity` works correctly without any
    additional normalisation step.

    :meth:`fit` is a no-op — the model is pre-trained and does not need to
    learn from the RLIP corpus.  It is provided to satisfy the
    :class:`~base.BaseEncoder` interface.

    Parameters
    ----------
    model_name:
        Any model identifier accepted by ``sentence_transformers.SentenceTransformer``.
        Defaults to ``"all-MiniLM-L6-v2"`` — a compact, general-purpose model
        well suited for semantic similarity.
    device:
        Inference device string passed to SentenceTransformer (e.g. ``"cpu"``,
        ``"cuda"``).  *None* lets the library auto-detect.

    Raises
    ------
    ImportError
        At encode time if ``sentence-transformers`` is not installed.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    # ── Private ───────────────────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceEncoder.\n"
                "Install it with:  pip install sentence-transformers\n"
                "Or choose a corpus-based encoder: TFIDFEncoder or BM25Encoder."
            ) from exc
        kwargs: dict[str, Any] = {}
        if self._device is not None:
            kwargs["device"] = self._device
        self._model = SentenceTransformer(self._model_name, **kwargs)

    # ── Public ────────────────────────────────────────────────────────────────

    def fit(self, corpus: list[str]) -> "SentenceEncoder":
        """
        No-op — the sentence-transformer model is pre-trained.

        Calling this method ensures the model is loaded (triggering a one-time
        download if needed) so subsequent :meth:`encode` calls are fast.

        Parameters
        ----------
        corpus:
            Accepted for API compatibility; not used.

        Returns
        -------
        self
        """
        self._ensure_model()
        return self

    def encode(self, text: str) -> np.ndarray:
        """
        Return the L2-normalised sentence embedding for *text*.

        Parameters
        ----------
        text:
            Any string.

        Returns
        -------
        np.ndarray
            1-D float64 array of length equal to the model's output dimension
            (384 for ``all-MiniLM-L6-v2``).
        """
        self._ensure_model()
        emb = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float64)
