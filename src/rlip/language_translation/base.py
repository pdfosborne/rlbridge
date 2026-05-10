"""
Base class for RLIP language translators.
"""

from __future__ import annotations

import abc
from typing import Any


class LanguageTranslator(abc.ABC):
    """
    Abstract base for environment-state → natural-language translators.

    Implementors receive the raw state (whatever the environment's
    ``step()`` / ``reset()`` returns as ``observation``) plus optional
    context and produce a human-readable description.

    Parameters passed to ``translate`` are all optional; translators should
    handle missing context gracefully.
    """

    #: Short human-readable name for this translator (override in subclasses).
    name: str = "base"

    @abc.abstractmethod
    def translate(
        self,
        state: Any,
        *,
        legal_moves: list[Any] | None = None,
        action_history: list[Any] | None = None,
    ) -> str:
        """
        Convert *state* into a natural-language description.

        Parameters
        ----------
        state:
            The raw observation returned by the environment.
        legal_moves:
            Optional list of currently legal action indices.
        action_history:
            Optional list of actions taken so far in the episode (oldest
            first).  Used to describe the most recent action taken.

        Returns
        -------
        str
            A complete English sentence (or paragraph) describing the state.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
