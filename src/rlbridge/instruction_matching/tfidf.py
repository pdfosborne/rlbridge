"""
TFIDFEncoder - bag-of-words TF-IDF instruction matcher.

This is the original encoder used in rlbridge, now exposed as a first-class
encoder type.  ``TextEncoder`` is kept as a backward-compatible alias.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .base import BaseEncoder


class TFIDFEncoder(BaseEncoder):
    """
    Bag-of-words TF-IDF text encoder backed by NumPy.

    Converts strings to L2-normalised TF-IDF vectors suitable for cosine-
    similarity comparison.  No external ML libraries are required beyond NumPy.

    The IDF formula follows the scikit-learn smooth convention::

        idf(t) = log((N + 1) / (df(t) + 1)) + 1

    where *N* is the corpus size and *df(t)* is the number of documents
    containing term *t*.

    Usage
    -----
    ::

        enc = TFIDFEncoder()
        enc.fit(corpus)                       # build vocab + IDF
        vec = enc.encode("some text")         # → 1-D numpy array (L2-normed)
        sim = enc.cosine_similarity(a, b)     # scalar in [-1, 1]
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._fitted: bool = False

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase alphabetic tokenisation (strips punctuation/numbers)."""
        return re.findall(r"[a-z]+", text.lower())

    # ── Public ────────────────────────────────────────────────────────────────

    def fit(self, corpus: list[str]) -> "TFIDFEncoder":
        """
        Build vocabulary and IDF weights from *corpus*.

        Parameters
        ----------
        corpus:
            List of text strings.  Should include the instruction *and* all
            observed language descriptions so that IDF weights are meaningful.

        Returns
        -------
        self
            Supports method chaining: ``enc = TFIDFEncoder().fit(corpus)``.
        """
        tokens_per_doc = [self._tokenize(t) for t in corpus]

        vocab: dict[str, int] = {}
        for tokens in tokens_per_doc:
            for tok in set(tokens):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self._vocab = vocab

        n = len(corpus)
        df = np.zeros(len(vocab), dtype=np.float64)
        for tokens in tokens_per_doc:
            for tok in set(tokens):
                idx = vocab.get(tok)
                if idx is not None:
                    df[idx] += 1.0

        # Smooth IDF (scikit-learn convention)
        self._idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
        self._fitted = True
        return self

    def encode(self, text: str) -> np.ndarray:
        """
        Return the L2-normalised TF-IDF vector for *text*.

        Out-of-vocabulary tokens are silently ignored; the resulting vector
        still captures the in-vocabulary signal.

        Parameters
        ----------
        text:
            Any string; tokenised internally.

        Returns
        -------
        np.ndarray
            1-D float64 array of length ``len(vocab)``.  The zero vector is
            returned when no known tokens are present.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before encode().")

        tokens = self._tokenize(text)
        vec = np.zeros(len(self._vocab), dtype=np.float64)
        if tokens:
            for tok in tokens:
                idx = self._vocab.get(tok)
                if idx is not None:
                    vec[idx] += 1.0
            vec /= len(tokens)           # normalised TF

        tfidf = vec * self._idf          # type: ignore[operator]
        norm = float(np.linalg.norm(tfidf))
        return tfidf / norm if norm > 0.0 else tfidf


# Backward-compatibility alias - existing code that imports TextEncoder keeps working.
TextEncoder = TFIDFEncoder
