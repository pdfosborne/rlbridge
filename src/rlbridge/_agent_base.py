"""
Shared base classes and helpers for rlbridge agents.
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


def _discrete_n(env: Any) -> Optional[int]:
    """Return the size of a fixed Discrete action space, or ``None``.

    ``None`` signals a variable / text action space (e.g. an rlbridge
    ``TextSpace``) whose number of valid actions changes per state and must be
    read from ``observation["legal_actions"]`` instead.
    """
    space = getattr(env, "action_space", None)
    if space is None:
        return None
    if getattr(space, "type", None) == "Discrete":
        return int(space.n)
    if isinstance(space, dict) and space.get("type") == "Discrete":
        return int(space["n"])
    if hasattr(space, "n"):  # Gymnasium Discrete space
        try:
            return int(space.n)
        except Exception:
            return None
    return None


def _n_actions_of(env: Any) -> int:
    """Infer the number of discrete actions from an rlbridge environment."""
    n = _discrete_n(env)
    return n if n is not None else 2  # safe fallback


def _legal_actions_of(obs: Any) -> Optional[list]:
    """Extract the ordered ``legal_actions`` list from an observation, if any.

    Environments with variable action spaces expose the actions that are valid
    in the current state under ``observation["legal_actions"]``. The list is
    index-aligned: selecting index ``i`` corresponds to ``legal_actions[i]``.
    """
    if isinstance(obs, dict):
        legal = obs.get("legal_actions")
        if isinstance(legal, (list, tuple)):
            return list(legal)
    return None


def _n_legal_of(obs: Any) -> Optional[int]:
    """Number of legal actions in *obs* (or ``None`` if not exposed)."""
    legal = _legal_actions_of(obs)
    return len(legal) if legal is not None else None


def _to_env_action(obs: Any, index: int, use_masking: bool) -> Any:
    """Map a policy action *index* to the action object the env expects.

    For variable action spaces the policy selects an index into the current
    ``legal_actions`` list; the environment is fed the corresponding action so
    that index ``i`` reliably resolves to ``legal_actions[i]``. For fixed
    Discrete spaces the integer index is returned unchanged.
    """
    if use_masking:
        legal = _legal_actions_of(obs)
        if legal and 0 <= index < len(legal):
            return legal[index]
    return index


_ACTION_CAPACITY_HEADROOM = 8


def _probe_max_legal(env: Any, probe_steps: int, seed: Optional[int]) -> int:
    """Roll out random legal moves to estimate the max legal-action count."""
    import random as _random

    rng = _random.Random(seed)
    max_n = 0
    try:
        reset_out = env.reset(seed=seed)
        obs = _get(reset_out, "observation", reset_out)
        for _ in range(max(1, probe_steps)):
            legal = _legal_actions_of(obs)
            if not legal:
                break
            max_n = max(max_n, len(legal))
            choice = rng.randrange(len(legal))
            step_out = env.step(legal[choice])
            obs = _get(step_out, "observation", obs)
            done = bool(_get(step_out, "terminated", False)) or bool(_get(step_out, "truncated", False))
            if done:
                reset_out = env.reset(seed=seed)
                obs = _get(reset_out, "observation", reset_out)
    except Exception:
        pass
    return max_n


def _infer_action_capacity(
    env: Any,
    *,
    probe_steps: int = 64,
    seed: Optional[int] = None,
) -> tuple[int, bool]:
    """Return ``(capacity, use_masking)`` for an environment.

    Fixed Discrete spaces return their exact size and disable masking. Variable
    / text action spaces are probed to size an action head large enough to
    cover the most legal actions ever offered (plus headroom); masking is then
    used to restrict the agent to the legal actions of each state.
    """
    n = _discrete_n(env)
    if n is not None:
        return n, False
    max_legal = _probe_max_legal(env, probe_steps, 0 if seed is None else int(seed))
    if max_legal <= 0:
        return _n_actions_of(env), False
    return max_legal + _ACTION_CAPACITY_HEADROOM, True


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
    if isinstance(obs, dict):
        out: list[float] = []
        # Sort keys for deterministic ordering across calls.
        for k in sorted(obs.keys(), key=lambda x: str(x)):
            out.extend(_flat_obs(str(k)))
            out.extend(_flat_obs(obs[k]))
        return out
    if obs is None:
        return [0.0]
    if isinstance(obs, bool):
        return [1.0 if obs else 0.0]
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
    """Base training result shared by all rlbridge agents."""

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
    """Minimal interface every rlbridge agent must satisfy."""

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
