"""
BaseEncoder — abstract interface for instruction-matching text encoders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEncoder(ABC):
    """
    Common interface for all instruction-matching text encoders.

    Subclasses must implement :meth:`fit` and :meth:`encode`.
    :meth:`cosine_similarity` is provided as a concrete static method and works
    correctly as long as :meth:`encode` returns L2-normalised vectors.

    Typical usage
    -------------
    ::

        enc = TFIDFEncoder()                        # or BM25Encoder(), etc.
        enc.fit([instruction] + language_corpus)    # build internal state
        q = enc.encode(instruction)
        scores = [(lang, enc.cosine_similarity(q, enc.encode(lang)))
                  for lang in language_corpus]
    """

    @abstractmethod
    def fit(self, corpus: list[str]) -> "BaseEncoder":
        """
        Build or adapt the encoder from a text corpus.

        For corpus-dependent encoders (TF-IDF, BM25) this builds vocabulary
        and statistics.  For pre-trained encoders (e.g. sentence-transformers)
        this is a no-op that returns *self* immediately.

        Parameters
        ----------
        corpus:
            List of text strings.  Should include both the instruction *and*
            all observed language descriptions so that IDF weights are
            meaningful for corpus-dependent methods.

        Returns
        -------
        self
            Supports method chaining: ``enc = TFIDFEncoder().fit(corpus)``.
        """

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """
        Return an L2-normalised embedding vector for *text*.

        Out-of-distribution tokens are handled gracefully by each
        implementation (silently ignored for bag-of-words methods; handled
        by the pretrained model for neural methods).

        Parameters
        ----------
        text:
            Any string.

        Returns
        -------
        np.ndarray
            1-D float64 array.  The zero vector is returned when *text*
            maps to nothing in the encoder's vocabulary (bag-of-words only).

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two vectors.

        When both *a* and *b* are L2-normalised (as produced by
        :meth:`encode`) this reduces to a simple dot product, which is
        numerically efficient and equivalent to cosine similarity.

        Parameters
        ----------
        a, b:
            1-D numpy arrays of the same length.

        Returns
        -------
        float
            Value in [-1, 1]; 1.0 = identical direction, 0.0 = orthogonal.
        """
        return float(np.dot(a, b))
