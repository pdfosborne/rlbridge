"""
rlip.instruction_matching
=========================
Text encoders for instruction-to-state matching in RLIP.

All encoders share the :class:`~base.BaseEncoder` interface:

* :meth:`~base.BaseEncoder.fit` — build internal state from a text corpus
* :meth:`~base.BaseEncoder.encode` — convert a string to an L2-normalised vector
* :meth:`~base.BaseEncoder.cosine_similarity` — dot product of two L2-normed vectors

Available encoders
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Class
     - Alias
     - Description
   * - :class:`TFIDFEncoder`
     - ``TextEncoder``
     - Bag-of-words TF-IDF (smooth IDF, NumPy only).  Default.
   * - :class:`BM25Encoder`
     -
     - Okapi BM25 with TF saturation and length normalisation (NumPy only).
   * - :class:`SentenceEncoder`
     -
     - Dense semantic embeddings via ``sentence-transformers`` (optional dep).

Quick start
-----------
::

    from rlip.instruction_matching import TFIDFEncoder, BM25Encoder, get_encoder

    # Explicit instantiation
    enc = BM25Encoder()
    enc.fit(corpus)
    sim = enc.cosine_similarity(enc.encode("sail to beach"), enc.encode("navigate shore"))

    # Factory lookup by name
    enc = get_encoder("bm25")
    enc.fit(corpus)

    # Use in match_instruction
    from rlip.instruction_following import match_instruction
    match = match_instruction("sail to beach", env, encoder=BM25Encoder())
"""

from .base import BaseEncoder
from .bm25 import BM25Encoder
from .sentence_transformer import SentenceEncoder
from .tfidf import TextEncoder, TFIDFEncoder

_REGISTRY: dict[str, type[BaseEncoder]] = {
    "tfidf": TFIDFEncoder,
    "bm25": BM25Encoder,
    "sentence-transformers": SentenceEncoder,
    "sentence_transformers": SentenceEncoder,
    "sentence": SentenceEncoder,
}


def get_encoder(name: str = "tfidf") -> BaseEncoder:
    """
    Return a fresh, unfitted encoder instance by name.

    Parameters
    ----------
    name:
        Encoder identifier.  Case-insensitive.  Supported values:

        * ``"tfidf"`` (default) — :class:`TFIDFEncoder`
        * ``"bm25"`` — :class:`BM25Encoder`
        * ``"sentence-transformers"`` / ``"sentence"`` — :class:`SentenceEncoder`

    Returns
    -------
    BaseEncoder
        A new, unfitted encoder instance.  Call :meth:`~BaseEncoder.fit`
        before :meth:`~BaseEncoder.encode`.

    Raises
    ------
    ValueError
        If *name* is not recognised.

    Example
    -------
    ::

        enc = get_encoder("bm25")
        enc.fit([instruction] + observed_descriptions)
        vec = enc.encode(instruction)
    """
    key = name.lower().replace(" ", "-")
    if key not in _REGISTRY:
        available = sorted({k for k in _REGISTRY if "_" not in k})
        raise ValueError(
            f"Unknown encoder {name!r}. Available names: {available}"
        )
    return _REGISTRY[key]()


__all__ = [
    "BaseEncoder",
    "TFIDFEncoder",
    "TextEncoder",   # backward-compat alias for TFIDFEncoder
    "BM25Encoder",
    "SentenceEncoder",
    "get_encoder",
]
