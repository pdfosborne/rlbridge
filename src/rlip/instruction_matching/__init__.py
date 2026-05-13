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


def get_encoder(
    name: str = "tfidf",
    *,
    sentence_model: str | None = None,
    sentence_device: str | None = None,
) -> BaseEncoder:
    """
    Return a fresh, unfitted encoder instance by name.

    Parameters
    ----------
    name:
        Encoder identifier.  Case-insensitive.  Supported values:

        * ``"tfidf"`` (default) — :class:`TFIDFEncoder`
        * ``"bm25"`` — :class:`BM25Encoder`
        * ``"sentence-transformers"`` / ``"sentence"`` — :class:`SentenceEncoder`
        * ``"sentence:<huggingface-model-id>"`` — :class:`SentenceEncoder`
          with a custom model, e.g. ``"sentence:BAAI/bge-small-en-v1.5"``.
    sentence_model:
        Optional explicit Hugging Face model ID for SentenceEncoder.
        Ignored for non-sentence encoders.
    sentence_device:
        Optional inference device (e.g. ``"cpu"``, ``"cuda"``) for
        SentenceEncoder. Ignored for non-sentence encoders.

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
    raw_name = name.strip()
    key = raw_name.lower().replace(" ", "-")

    # Allow inline model spec, e.g. "sentence:BAAI/bge-small-en-v1.5".
    inline_model: str | None = None
    if ":" in raw_name:
        prefix, suffix = raw_name.split(":", 1)
        pkey = prefix.strip().lower().replace(" ", "-")
        if pkey in {"sentence", "sentence-transformers", "sentence_transformers", "hf"}:
            key = "sentence"
            inline_model = suffix.strip() or None

    if key not in _REGISTRY:
        available = sorted({k for k in _REGISTRY if "_" not in k})
        raise ValueError(
            f"Unknown encoder {name!r}. Available names: {available}"
        )

    encoder_cls = _REGISTRY[key]
    if encoder_cls is SentenceEncoder:
        model_name = sentence_model or inline_model or "all-MiniLM-L6-v2"
        return SentenceEncoder(model_name=model_name, device=sentence_device)
    return encoder_cls()


__all__ = [
    "BaseEncoder",
    "TFIDFEncoder",
    "TextEncoder",   # backward-compat alias for TFIDFEncoder
    "BM25Encoder",
    "SentenceEncoder",
    "get_encoder",
]
