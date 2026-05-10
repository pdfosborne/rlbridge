"""
Shared base classes and helpers for RLIP agents.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Observation / action helpers ──────────────────────────────────────────────

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Dict- or attribute-access, mirrors interaction_protocols._get."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _n_actions_of(env: Any) -> int:
    """Infer the number of discrete actions from an RLIP environment."""
    space = getattr(env, "action_space", None)
    if space is None:
        return 2  # safe fallback
    space_type = getattr(space, "type", None)

    if space_type == "Discrete":
        return int(space.n)
    if isinstance(space, dict) and space.get("type") == "Discrete":
        return int(space["n"])
    # Gymnasium Discrete space
    if hasattr(space, "n"):
        return int(space.n)
    return 2


def _env_id_of(env: Any) -> str:
    return getattr(env, "env_id", type(env).__name__)


def _flat_obs(obs: Any) -> list[float]:
    """
    Flatten any observation type to a plain Python list of floats.

    Handles: str (hashed to a single float), list/tuple, numpy array.
    """
    if isinstance(obs, str):
        # Deterministic hash → float in [0, 1) via simple FNV-style fold
        h = 0
        for ch in obs.encode():
            h = (h * 31 + ch) & 0xFFFFFFFF
        return [h / 0xFFFFFFFF]
    if isinstance(obs, (int, float)):
        return [float(obs)]
    if isinstance(obs, (list, tuple)):
        out: list[float] = []
        for v in obs:
            out.extend(_flat_obs(v))
        return out
    try:
        import numpy as np  # noqa: PLC0415
        if isinstance(obs, np.ndarray):
            return obs.flatten().tolist()
    except ImportError:
        pass
    return [float(hash(obs) & 0xFFFF) / 0xFFFF]


# ── Shared train-result base ──────────────────────────────────────────────────

@dataclass
class TrainResult:
    """Base training result shared by all RLIP agents."""

    agent_name: str
    n_episodes: int
    episode_rewards: list[float] = field(default_factory=list, repr=False)
    final_epsilon: float = 0.0
    best_episode_history: list[tuple[Any, Any]] = field(default_factory=list, repr=False)
    """(obs, action) pairs from the highest-reward training episode.
    Captured during training (where ε-greedy finds good paths) so it can be
    replayed directly without re-running the greedy policy."""

    @property
    def mean_reward(self) -> float:
        if not self.episode_rewards:
            return 0.0
        return sum(self.episode_rewards) / len(self.episode_rewards)

    @property
    def best_reward(self) -> float:
        return max(self.episode_rewards) if self.episode_rewards else 0.0

    @property
    def last_n_mean(self) -> float:
        """Mean reward over the last 10 % of episodes (at least 1)."""
        n = max(1, len(self.episode_rewards) // 10)
        tail = self.episode_rewards[-n:]
        return sum(tail) / len(tail) if tail else 0.0


# ── Abstract base ─────────────────────────────────────────────────────────────

class AgentBase(abc.ABC):
    """Minimal interface every RLIP agent must satisfy."""

    name: str = "base"

    @abc.abstractmethod
    def act(self, obs: Any) -> Any:
        """Return an action for the given observation (greedy / no exploration)."""

    @abc.abstractmethod
    def train(self, env: Any, n_episodes: int, **kwargs: Any) -> TrainResult:
        """Train the agent on *env* for *n_episodes* episodes."""

    @abc.abstractmethod
    def save(self, path: Any) -> None:
        """Persist the agent's learned parameters to *path*."""

    @abc.abstractmethod
    def load(self, path: Any) -> None:
        """Restore the agent's learned parameters from *path*."""

    def __repr__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        kv = ", ".join(f"{k}={v!r}" for k, v in list(attrs.items())[:6])
        return f"{type(self).__name__}({kv})"
