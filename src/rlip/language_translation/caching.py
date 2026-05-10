"""
CachingTranslator — memoised wrapper around any LanguageTranslator.

Observations are often visited many times across training episodes.  This
module provides a transparent cache layer: the first ``translate()`` call for
a given observation stores the result; all subsequent calls for the same
observation return the stored string without re-running the underlying
translator logic.

The cache is global and keyed by ``env_id`` so that different environments
never share entries, but observations from the *same* environment across
multiple instruction-matching sessions are shared automatically.

Usage
-----
::

    from rlip.language_translation import SailingLanguageTranslator
    from rlip.language_translation.caching import CachingTranslator

    inner = SailingLanguageTranslator()
    cached = CachingTranslator(inner, env_id="Sailing-v0")

    desc = cached.translate(obs)   # computed + stored
    desc = cached.translate(obs)   # returned from cache
"""

from __future__ import annotations

from typing import Any

from .base import LanguageTranslator

# ── Module-level cache ────────────────────────────────────────────────────────
# Keyed by env_id → { obs_key → language_string }
_TRANSLATION_CACHE: dict[str, dict[str, str]] = {}


def _obs_key(obs: Any) -> str:
    """
    Derive a stable string key for an observation.

    * NumPy arrays → hex of raw bytes + shape (unique for any array value).
    * Other types  → ``repr(obs)`` (works for scalars, strings, tuples, etc.).
    """
    try:
        import numpy as np  # noqa: PLC0415
        if isinstance(obs, np.ndarray):
            return f"{obs.shape}:{obs.tobytes().hex()}"
    except ImportError:
        pass
    return repr(obs)


def clear_translation_cache(env_id: str | None = None) -> None:
    """
    Remove entries from the global translation cache.

    Parameters
    ----------
    env_id:
        Clear only entries for this environment.  When *None* the entire
        cache is cleared.

    Example
    -------
    ::

        from rlip.language_translation.caching import clear_translation_cache
        clear_translation_cache("Sailing-v0")   # clear one env
        clear_translation_cache()               # clear all
    """
    if env_id is None:
        _TRANSLATION_CACHE.clear()
    else:
        _TRANSLATION_CACHE.pop(env_id, None)


def translation_cache_info(env_id: str | None = None) -> dict[str, int]:
    """
    Return the number of cached translations per environment.

    Parameters
    ----------
    env_id:
        Restrict to a single environment.  When *None* all environments are
        included.

    Returns
    -------
    dict[str, int]
        Mapping of ``env_id → number_of_cached_entries``.
    """
    if env_id is not None:
        return {env_id: len(_TRANSLATION_CACHE.get(env_id, {}))}
    return {k: len(v) for k, v in _TRANSLATION_CACHE.items()}


# ── CachingTranslator ─────────────────────────────────────────────────────────

class CachingTranslator(LanguageTranslator):
    """
    Transparent memoising wrapper for any :class:`LanguageTranslator`.

    Caches ``translate()`` results in the module-level
    :data:`_TRANSLATION_CACHE` dict, keyed by ``env_id`` and a
    deterministic observation key.

    .. note::
        Caching is keyed *only* on the observation value, not on
        ``legal_moves`` or ``action_history``.  If your translator produces
        meaningfully different output depending on those context arguments
        you should disable caching for that translator or subclass and
        override ``_cache_key``.

    Parameters
    ----------
    translator:
        The underlying translator to delegate uncached calls to.
    env_id:
        Environment identifier used as the cache namespace.  Should match
        the value stored in ``env.env_id``.
    """

    def __init__(self, translator: LanguageTranslator, env_id: str) -> None:
        self._translator = translator
        self._env_id = env_id
        self.name = f"caching:{translator.name}"

    # ── Public ────────────────────────────────────────────────────────────────

    def translate(
        self,
        state: Any,
        *,
        legal_moves: list[Any] | None = None,
        action_history: list[Any] | None = None,
    ) -> str:
        """
        Return the language description of *state*, from cache when possible.

        On a cache miss the underlying translator is called and the result is
        stored before being returned.

        Parameters
        ----------
        state:
            Raw environment observation.
        legal_moves:
            Passed through to the underlying translator on a cache miss.
        action_history:
            Passed through to the underlying translator on a cache miss.

        Returns
        -------
        str
        """
        env_cache = _TRANSLATION_CACHE.setdefault(self._env_id, {})
        key = _obs_key(state)
        if key not in env_cache:
            env_cache[key] = self._translator.translate(
                state,
                legal_moves=legal_moves,
                action_history=action_history,
            )
        return env_cache[key]

    def cache_size(self) -> int:
        """Number of entries cached for this environment."""
        return len(_TRANSLATION_CACHE.get(self._env_id, {}))

    def __repr__(self) -> str:
        return (
            f"CachingTranslator(translator={self._translator!r}, "
            f"env_id={self._env_id!r}, cached={self.cache_size()})"
        )
