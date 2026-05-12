"""
RLIP Language Translation
==========================
Maps raw RL environment observations (states) to natural-language descriptions.

Each translator is a callable that accepts the raw state plus optional context
(legal moves, action history) and returns a plain-English string.

Usage
-----
    from rlip.language_translation.sailing import SailingLanguageTranslator

    translator = SailingLanguageTranslator()
    text = translator.translate("0.0300_0.2", action_history=[1, 0])
    # → "The boat is in the middle, close hauled with wind on the port side,
    #    the last action was to turn towards the beach."

Extending
---------
Sub-class :class:`LanguageTranslator` and implement ``translate()``, then
register the instance in ``TRANSLATORS`` at the bottom of this file or in
your own module.
"""

from .base import LanguageTranslator
from .caching import (
    CachingTranslator,
    clear_translation_cache,
    translation_cache_info,
)
from .sailing import SailingLanguageTranslator
from .textworld import TextWorldTranslator
from .generator import (
    LLMCallable,
    GeneratedTranslator,
    TranslatorGenerator,
    build_translator,
)

#: Registry of built-in translators keyed by their ``env_id`` prefix.
TRANSLATORS: dict[str, LanguageTranslator] = {
    "Sailing-v0":                SailingLanguageTranslator(),
    "Sailing-Hard-v0":           SailingLanguageTranslator(),
    # Direct class-name fallback for SailingEnvironment used outside the registry
    "SailingEnvironment":        SailingLanguageTranslator(),
    # TextWorld text-adventure environments (observation is already language)
    "TextWorld-Take-v0":                   TextWorldTranslator(),
    "TextWorld-Navigate-v0":               TextWorldTranslator(),
    "TextWorld-TreasureHunt-v0":           TextWorldTranslator(),
    "TextWorld-CoinCollector-Easy-v0":     TextWorldTranslator(),
    "TextWorld-CoinCollector-Medium-v0":   TextWorldTranslator(),
    "TextWorld-Cooking-Easy-v0":           TextWorldTranslator(),
    "TextWorld-Cooking-Medium-v0":         TextWorldTranslator(),
    # Chess — observation is already descriptive text (board + legal moves)
    "Chess-v0":                            None,  # passthrough
    "Chess-SelfPlay-v0":                   None,  # passthrough
    # Pokemon Red — observation is already descriptive text from memory
    "PokemonRed-Gary-Battle-v0":           None,  # passthrough: obs is already text
}


def get_translator(env_id: str) -> LanguageTranslator | None:
    """
    Return the :class:`LanguageTranslator` registered for *env_id*, or
    ``None`` if no translator is registered.
    """
    return TRANSLATORS.get(env_id)


def translate(
    env_id: str,
    state: object,
    *,
    legal_moves: list | None = None,
    action_history: list | None = None,
) -> str | None:
    """
    Convenience wrapper — translate *state* for *env_id*.

    Returns ``None`` if no translator is registered for that environment.
    """
    translator = get_translator(env_id)
    if translator is None:
        return None
    return translator.translate(
        state,
        legal_moves=legal_moves,
        action_history=action_history,
    )


__all__ = [
    "LanguageTranslator",
    "SailingLanguageTranslator",
    "TextWorldTranslator",
    "CachingTranslator",
    "TRANSLATORS",
    "get_translator",
    "translate",
    "clear_translation_cache",
    "translation_cache_info",
    # Generator
    "LLMCallable",
    "GeneratedTranslator",
    "TranslatorGenerator",
    "build_translator",
]
