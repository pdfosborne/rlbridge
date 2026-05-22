"""
Abstract base class for rlbridge environment adapters.

Any RL environment can be wrapped by sub-classing rlbridgeEnvironment and
implementing the abstract methods.  The Gymnasium adapter in
`gymnasium_adapter.py` is the reference implementation.
"""

from __future__ import annotations

import abc
from typing import Any, Optional

from ..protocol.messages import (
    EnvironmentInfo,
    ResetResult,
    StepResult,
    RenderResult,
    SpaceDescription,
)


class rlbridgeEnvironment(abc.ABC):
    """Protocol-level wrapper around a single RL environment instance."""

    # ── Life-cycle ───────────────────────────────────────────────────────────

    @abc.abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        """Reset the environment and return the initial observation."""

    @abc.abstractmethod
    def step(self, action: Any) -> StepResult:
        """Execute one action and return the transition tuple."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release all resources held by this environment instance."""

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def observation_space(self) -> SpaceDescription:
        """Serialisable description of the observation space."""

    @property
    @abc.abstractmethod
    def action_space(self) -> SpaceDescription:
        """Serialisable description of the action space."""

    @abc.abstractmethod
    def render(self) -> RenderResult:
        """Render the current state and return a RenderResult."""

    # ── Optional ─────────────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        """True if reset() has been called at least once."""
        return getattr(self, "_initialized", False)

    def sample_action(self) -> Any:
        """Return a random valid action (useful for agents that want to explore)."""
        raise NotImplementedError("sample_action is not implemented by this adapter")


class rlbridgeEnvironmentFactory(abc.ABC):
    """
    Factory that the registry uses to instantiate environments.

    Implementors register themselves with :class:`EnvironmentRegistry`.
    """

    @property
    @abc.abstractmethod
    def env_info(self) -> EnvironmentInfo:
        """Static metadata for this environment type."""

    @abc.abstractmethod
    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> rlbridgeEnvironment:
        """Instantiate a new :class:`rlbridgeEnvironment`."""
