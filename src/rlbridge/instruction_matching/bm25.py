"""
BM25Encoder - Okapi BM25 instruction matcher.

Okapi BM25 is a probabilistic ranking function that improves on TF-IDF in
two important ways:

* **TF saturation** - Repeatedly mentioning a word has diminishing returns,
  preventing long documents from unfairly dominating.
* **Length normalisation** - Scores are adjusted for document length relative
  to the corpus average, so short and long descriptions are compared fairly.

This encoder represents each document as a BM25-weighted term vector (one
dimension per vocabulary term), then L2-normalises it so that standard cosine
similarity can be used - the same interface as :class:`~tfidf.TFIDFEncoder`.

Robertson's smoothed IDF formula is used::

    idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

BM25 TF saturation::

    tf_sat(t, d) = tf(t,d) * (k1 + 1)
                   ─────────────────────────────────────────────
                   tf(t,d) + k1 * (1 - b + b * len(d) / avgdl)

Default hyperparameters follow the commonly recommended values:
* ``k1 = 1.5`` - TF saturation ceiling
* ``b  = 0.75`` - length normalisation strength (0 = off, 1 = full)
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .base import BaseEncoder


class BM25Encoder(BaseEncoder):
    """
    Okapi BM25 text encoder.

    Better suited than TF-IDF for instruction matching because it penalises
    high term frequency (saturation) and normalises for text length, making
    it more robust when language descriptions vary in verbosity.

    Parameters
    ----------
    k1:
        TF saturation parameter.  Higher values give more weight to repeated
        terms before they plateau.  Typical range: 1.2–2.0.
    b:
        Length normalisation parameter.  0.0 disables length normalisation
        (pure TF-IDF-like); 1.0 applies full normalisation.

    Usage
    -----
    ::

        enc = BM25Encoder()
        enc.fit(corpus)
        q_vec   = enc.encode(query)
        doc_vec = enc.encode(doc)
        sim = enc.cosine_similarity(q_vec, doc_vec)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._vocab: dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._avgdl: float = 0.0
        self._fitted: bool = False

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase alphabetic tokenisation (strips punctuation/numbers)."""
        return re.findall(r"[a-z]+", text.lower())

    # ── Public ────────────────────────────────────────────────────────────────

    def fit(self, corpus: list[str]) -> "BM25Encoder":
        """
        Build vocabulary, IDF weights, and average document length from *corpus*.

        Parameters
        ----------
        corpus:
            List of text strings (instruction + language descriptions).

        Returns
        -------
        self
        """
        tokens_per_doc = [self._tokenize(t) for t in corpus]

        # Build vocabulary
        vocab: dict[str, int] = {}
        for tokens in tokens_per_doc:
            for tok in set(tokens):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self._vocab = vocab

        n = len(corpus)
        lengths = np.array([len(t) for t in tokens_per_doc], dtype=np.float64)
        self._avgdl = float(lengths.mean()) if n > 0 else 1.0

        # Document frequencies
        df = np.zeros(len(vocab), dtype=np.float64)
        for tokens in tokens_per_doc:
            for tok in set(tokens):
                idx = vocab.get(tok)
                if idx is not None:
                    df[idx] += 1.0

        # Robertson's smoothed IDF
        self._idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0)
        self._fitted = True
        return self

    def encode(self, text: str) -> np.ndarray:
        """
        Return the L2-normalised BM25 vector for *text*.

        Each dimension corresponds to a vocabulary term weighted by
        ``idf(t) * tf_sat(t, text)``.  The vector is L2-normalised so that
        :meth:`~base.BaseEncoder.cosine_similarity` gives a value in [-1, 1].

        Parameters
        ----------
        text:
            Any string.

        Returns
        -------
        np.ndarray
            1-D float64 array of length ``len(vocab)``.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before encode().")

        tokens = self._tokenize(text)
        doc_len = len(tokens)

        # Raw term frequencies
        tf_raw = np.zeros(len(self._vocab), dtype=np.float64)
        for tok in tokens:
            idx = self._vocab.get(tok)
            if idx is not None:
                tf_raw[idx] += 1.0

        # BM25 TF saturation
        k1, b = self.k1, self.b
        denom = tf_raw + k1 * (1.0 - b + b * doc_len / max(self._avgdl, 1.0))
        # Avoid division by zero on terms with zero TF
        with np.errstate(divide="ignore", invalid="ignore"):
            tf_sat = np.where(denom > 0.0, tf_raw * (k1 + 1.0) / denom, 0.0)

        bm25_vec = tf_sat * self._idf  # type: ignore[operator]
        norm = float(np.linalg.norm(bm25_vec))
        return bm25_vec / norm if norm > 0.0 else bm25_vec
