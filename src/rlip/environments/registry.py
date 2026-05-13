"""
Environment Registry
=====================
Central registry mapping environment IDs to their factories.

Usage
-----
    from rlip.environments.registry import registry

    # Register a custom factory
    registry.register(MyFactory())

    # Look up a factory
    factory = registry.get("CartPole-v1")
    env = factory.create(render_mode="rgb_array")
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from ..protocol.messages import EnvironmentInfo
from .base import RLIPEnvironment, RLIPEnvironmentFactory


class EnvironmentRegistry:
    """Thread-safe registry of :class:`RLIPEnvironmentFactory` instances."""

    def __init__(self) -> None:
        self._factories: dict[str, RLIPEnvironmentFactory] = {}
        self._lock = threading.RLock()

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, factory: RLIPEnvironmentFactory) -> None:
        """Register a factory under its ``env_info.env_id``."""
        with self._lock:
            self._factories[factory.env_info.env_id] = factory

    def register_gymnasium_ids(
        self,
        env_ids: list[str],
        *,
        tags: Optional[list[str]] = None,
    ) -> None:
        """
        Convenience method: batch-register Gymnasium environment IDs.
        Silently skips IDs that are not installed.
        """
        from .gymnasium_adapter import GymnasiumFactory

        with self._lock:
            for env_id in env_ids:
                try:
                    factory = GymnasiumFactory(env_id, tags=tags or [])
                    self._factories[env_id] = factory
                except Exception:
                    pass

    def auto_register_gymnasium(self) -> int:
        """
        Discover and register all Gymnasium environments available in the
        current Python environment.  Returns the number newly registered.
        """
        import gymnasium as gym
        from .gymnasium_adapter import GymnasiumFactory

        registered = 0
        with self._lock:
            for env_id in gym.envs.registry.keys():
                if env_id not in self._factories:
                    try:
                        factory = GymnasiumFactory(env_id)
                        self._factories[env_id] = factory
                        registered += 1
                    except Exception:
                        pass
        return registered

    # ── Look-up ───────────────────────────────────────────────────────────────

    def get(self, env_id: str) -> RLIPEnvironmentFactory:
        with self._lock:
            factory = self._factories.get(env_id)
        if factory is None:
            from ..protocol.constants import ErrorCodes
            from ..server.exceptions import RLIPError
            raise RLIPError(
                code=ErrorCodes.ENV_NOT_FOUND,
                message=f"Environment '{env_id}' is not registered",
                data={"env_id": env_id},
            )
        return factory

    def list_environments(
        self,
        tags: Optional[list[str]] = None,
        namespace: Optional[str] = None,
    ) -> list[EnvironmentInfo]:
        with self._lock:
            factories = list(self._factories.values())

        infos: list[EnvironmentInfo] = []
        for f in factories:
            info = f.env_info
            if namespace and info.namespace != namespace:
                continue
            if tags and not set(tags).issubset(set(info.tags)):
                continue
            infos.append(info)
        return infos

    def create(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> RLIPEnvironment:
        factory = self.get(env_id)
        return factory.create(render_mode=render_mode, **kwargs)

    def __len__(self) -> int:
        with self._lock:
            return len(self._factories)

    def __contains__(self, env_id: str) -> bool:
        with self._lock:
            return env_id in self._factories


# Global singleton registry, pre-populated with common Gymnasium envs
registry = EnvironmentRegistry()

# Register the most common classic-control environments immediately so the
# server works out of the box without calling auto_register_gymnasium().
_COMMON_ENVS = [
    "CartPole-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "Acrobot-v1",
    "LunarLander-v3",
    "LunarLanderContinuous-v3",
    "BipedalWalker-v3",
    "BipedalWalkerHardcore-v3",
    "CarRacing-v3",
]
registry.register_gymnasium_ids(_COMMON_ENVS)

# Register built-in sailing environments
from .predefined.sailing import ALL_SAILING_FACTORIES  # noqa: E402
for _factory in ALL_SAILING_FACTORIES:
    registry.register(_factory)

# Register built-in grid world environments
from .predefined.gridworld import ALL_GRIDWORLD_FACTORIES  # noqa: E402
for _factory in ALL_GRIDWORLD_FACTORIES:
    registry.register(_factory)

# Register Flesh and Blood environments
from .predefined.flesh_and_blood import ALL_FAB_FACTORIES  # noqa: E402
for _factory in ALL_FAB_FACTORIES:
    registry.register(_factory)

# Register TextWorld environments (optional dependency)
from .predefined import ALL_TEXTWORLD_FACTORIES as _tw_factories  # noqa: E402
for _factory in _tw_factories:
    registry.register(_factory)

# Register Chess environments (optional dependency)
from .predefined import ALL_CHESS_FACTORIES as _chess_factories  # noqa: E402
for _factory in _chess_factories:
    registry.register(_factory)

# Register Pokemon Red environments (optional dependency — requires PyBoy)
from .predefined import ALL_POKEMON_RED_FACTORIES as _pr_factories  # noqa: E402
for _factory in _pr_factories:
    registry.register(_factory)
