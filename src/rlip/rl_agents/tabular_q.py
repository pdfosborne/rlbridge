"""
Tabular Q-Learning Agent
=========================
Classic lookup-table Q-learning with ε-greedy exploration and optional
ε decay.  Works on any RLIP environment whose observations are hashable
(strings, ints, tuples) — ideal for text-observation envs like
``Sailing-v0``.

Algorithm
---------
For each step in each episode:

1. Choose action with ε-greedy: with probability ε pick uniformly at random,
   otherwise pick ``argmax_a Q(s, a)``.
2. Execute the action, observe ``(s', r, done)``.
3. Update::

       Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]

4. Decay ε: ``ε ← max(ε_min, ε · ε_decay)`` after each episode.

References
----------
Watkins & Dayan (1992), "Q-learning", Machine Learning.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .._agent_base import AgentBase, TrainResult, _get, _n_actions_of


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TabularQTrainResult(TrainResult):
    """Training summary for :class:`TabularQAgent`."""

    q_table_size: int = 0
    """Number of unique (state, action) pairs ever visited."""

    def __str__(self) -> str:
        return (
            f"TabularQTrainResult  episodes={self.n_episodes}"
            f"  mean_reward={self.mean_reward:.4f}"
            f"  best_reward={self.best_reward:.4f}"
            f"  q_table_size={self.q_table_size}"
            f"  final_epsilon={self.final_epsilon:.4f}"
        )


# ── Agent ─────────────────────────────────────────────────────────────────────

class TabularQAgent(AgentBase):
    """
    Tabular Q-learning agent.

    Parameters
    ----------
    n_actions:
        Size of the discrete action space.  Pass 0 to auto-detect from the
        environment on the first call to ``train()``.
    alpha:
        Learning rate (0, 1].
    gamma:
        Discount factor [0, 1].
    epsilon:
        Initial exploration rate.
    epsilon_min:
        Floor for ε after decay.
    epsilon_decay:
        Multiplicative decay applied to ε after each episode.
    q_init:
        Initial Q-value for unseen state–action pairs (optimistic: > 0,
        pessimistic: < 0, neutral: 0.0).
    seed:
        Random seed for reproducibility.
    """

    name = "tabular_q"

    def __init__(
        self,
        n_actions: int = 0,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        q_init: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_init = q_init
        self._rng = random.Random(seed)
        # Q-table: state → list of Q-values per action
        self._q: dict[Any, list[float]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def act(self, obs: Any) -> int:
        """Return the greedy action for *obs* (no exploration)."""
        return int(self._greedy_action(obs))

    def train(
        self,
        env: Any,
        n_episodes: int = 500,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ) -> TabularQTrainResult:
        """
        Run Q-learning training on *env*.

        Parameters
        ----------
        env:
            Any RLIP environment.
        n_episodes:
            Total number of training episodes.
        max_steps:
            Step cap per episode.
        seed:
            Base seed; each episode uses ``seed + episode_index`` when set.

        Returns
        -------
        TabularQTrainResult
        """
        if self.n_actions == 0:
            self.n_actions = _n_actions_of(env)

        episode_rewards: list[float] = []
        best_ep_reward  = float("-inf")
        best_ep_history: list[tuple] = []

        for ep in range(n_episodes):
            ep_seed = (seed + ep) if seed is not None else None
            reset_out = env.reset(seed=ep_seed)
            obs = _get(reset_out, "observation", reset_out)

            total_reward = 0.0
            ep_history: list[tuple] = []

            for _ in range(max_steps):
                action = self._epsilon_greedy(obs)
                ep_history.append((obs, action))
                step_out = env.step(action)
                next_obs  = _get(step_out, "observation", obs)
                reward     = float(_get(step_out, "reward", 0.0))
                terminated = bool(_get(step_out, "terminated", False))
                truncated  = bool(_get(step_out, "truncated", False))

                self._update(obs, action, reward, next_obs, terminated or truncated)
                total_reward += reward
                obs = next_obs

                if terminated or truncated:
                    break

            episode_rewards.append(total_reward)
            if total_reward > best_ep_reward:
                best_ep_reward  = total_reward
                best_ep_history = list(ep_history)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        n_pairs = sum(len(v) for v in self._q.values())
        return TabularQTrainResult(
            agent_name=self.name,
            n_episodes=n_episodes,
            episode_rewards=episode_rewards,
            final_epsilon=self.epsilon,
            best_episode_history=best_ep_history,
            q_table_size=n_pairs,
        )

    def save(self, path: str | Path) -> None:
        """Serialise the Q-table and hyper-parameters to a JSON file."""
        # Keys may be strings, ints, or tuples; store as tagged entries.
        entries = []
        for k, v in self._q.items():
            if isinstance(k, tuple):
                entries.append({"ktype": "tuple", "key": list(k), "qvals": v})
            elif isinstance(k, str):
                entries.append({"ktype": "str", "key": k, "qvals": v})
            else:
                entries.append({"ktype": "other", "key": k, "qvals": v})
        data = {
            "agent": self.name,
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q_init": self.q_init,
            "q_table": entries,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Restore Q-table and hyper-parameters from a JSON file."""
        data = json.loads(Path(path).read_text())
        self.n_actions    = data["n_actions"]
        self.alpha        = data["alpha"]
        self.gamma        = data["gamma"]
        self.epsilon      = data["epsilon"]
        self.epsilon_min  = data["epsilon_min"]
        self.epsilon_decay = data["epsilon_decay"]
        self.q_init       = data["q_init"]
        self._q = {}
        for entry in data["q_table"]:
            ktype = entry["ktype"]
            if ktype == "tuple":
                key: Any = tuple(entry["key"])
            else:
                key = entry["key"]
            self._q[key] = entry["qvals"]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _q_values(self, obs: Any) -> list[float]:
        key = _hashable(obs)
        if key not in self._q:
            self._q[key] = [self.q_init] * self.n_actions
        return self._q[key]

    def _greedy_action(self, obs: Any) -> int:
        qv = self._q_values(obs)
        best = max(qv)
        # Break ties randomly
        best_actions = [i for i, v in enumerate(qv) if v == best]
        return self._rng.choice(best_actions)

    def _epsilon_greedy(self, obs: Any) -> int:
        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.n_actions - 1)
        return self._greedy_action(obs)

    def _update(
        self,
        obs: Any,
        action: int,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        qv = self._q_values(obs)
        next_max = max(self._q_values(next_obs)) if not done else 0.0
        td_target = reward + self.gamma * next_max
        qv[action] += self.alpha * (td_target - qv[action])


# ── Helper ────────────────────────────────────────────────────────────────────

def _hashable(obs: Any) -> Any:
    """Convert an observation to a hashable dict key."""
    if isinstance(obs, dict):
        # Deterministic ordering for stable Q-table keys.
        return tuple(
            (str(k), _hashable(v))
            for k, v in sorted(obs.items(), key=lambda item: str(item[0]))
        )
    if isinstance(obs, (list, tuple)):
        return tuple(_hashable(v) for v in obs)
    if isinstance(obs, float):
        return round(obs, 6)
    if obs is None:
        return None
    try:
        import numpy as np  # noqa: PLC0415
        if isinstance(obs, np.ndarray):
            return tuple(_hashable(v) for v in obs.flatten().tolist())
    except ImportError:
        pass
    # Last resort: coerce unhashable objects to a stable string representation.
    try:
        hash(obs)
        return obs
    except TypeError:
        return repr(obs)
