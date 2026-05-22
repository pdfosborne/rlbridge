"""
TextWorld Language Translator
==============================
The ``observation`` in a TextWorld game is already natural language
(the text-adventure narrator's output).  This translator normalises it
by preferring the richer ``description`` + ``inventory`` fields that
TextWorld exposes via ``info``, and falls back to the raw observation
string when info is unavailable.

Usage
-----
Translators are called by :func:`rlbridge.instruction_following.match_instruction`
via the ``info`` dict:

::

    translator = TextWorldTranslator()
    # called with the raw obs (game output text)
    lang = translator.translate(obs_string, legal_moves=admissible_commands)
"""

from __future__ import annotations

import re
from typing import Any

from .base import LanguageTranslator

# TextWorld often puts a long ASCII-art banner at the start of the first obs.
# This regex strips the banner (runs of box-drawing / whitespace at the start).
_BANNER_RE = re.compile(r"^[\s\$\\|_\-/]+\n.*?\n\n", re.DOTALL)
# Strip "> " prompt lines at the end
_PROMPT_RE = re.compile(r"\n?\s*>\s*$")


def _clean(text: str) -> str:
    """Remove the TextWorld ASCII banner and trailing prompt from *text*."""
    text = _BANNER_RE.sub("", text)
    text = _PROMPT_RE.sub("", text)
    return text.strip()


class TextWorldTranslator(LanguageTranslator):
    """
    Identity-style translator for TextWorld text-adventure environments.

    Since TextWorld observations are already natural language, this
    translator simply cleans up the text and optionally appends the
    inventory to provide full context.
    """

    name = "textworld"

    def translate(
        self,
        state: Any,
        *,
        legal_moves: list[Any] | None = None,
        action_history: list[Any] | None = None,
    ) -> str:
        """
        Convert a TextWorld game observation to a normalised language string.

        Parameters
        ----------
        state:
            The raw game-output string returned by the environment's
            ``reset()`` or ``step()``.
        legal_moves:
            List of currently admissible command strings (optional).
            Not used in the output but accepted for API compatibility.
        action_history:
            Not used; accepted for API compatibility.

        Returns
        -------
        str
            The cleaned game text.  If *state* is a dict (future-proofing for
            richer info dicts), ``description`` and ``inventory`` are combined.
        """
        if isinstance(state, dict):
            parts: list[str] = []
            if state.get("description"):
                parts.append(state["description"].strip())
            if state.get("inventory"):
                inv = state["inventory"].strip()
                if inv and inv.lower() not in ("", "you are carrying nothing."):
                    parts.append(f"Inventory: {inv}")
            return " | ".join(parts) if parts else str(state)

        cleaned = _clean(str(state))
        return cleaned if cleaned else str(state)
