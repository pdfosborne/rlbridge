"""
Deep Q-Network (DQN) Agent
===========================
A from-scratch NumPy implementation of DQN with experience replay and a
target network.  No deep-learning framework is required.

Network architecture
--------------------
A two-hidden-layer fully-connected network with ReLU activations::

    obs_dim → hidden_size → hidden_size → n_actions

Weights are stored as plain NumPy arrays; forward and backward passes are
written explicitly.

Algorithm
---------
1. Initialise online network Q and target network Q̂ with identical weights.
2. Maintain a circular replay buffer of ``(s, a, r, s', done)`` transitions.
3. At each step:
   a. ε-greedy action selection using Q.
   b. Store transition in replay buffer.
   c. If buffer has ≥ ``batch_size`` samples, draw a random mini-batch and
      compute targets::

          y = r + γ · max_a' Q̂(s', a') · (1 − done)

   d. Update Q with one step of SGD to minimise ``(y − Q(s,a))²``.
4. Every ``target_update_freq`` steps copy Q → Q̂.
5. Decay ε: ``ε ← max(ε_min, ε · ε_decay)`` after each episode.

References
----------
Mnih et al. (2015), "Human-level control through deep reinforcement learning",
Nature 518.
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .._agent_base import AgentBase, TrainResult, _flat_obs, _get, _n_actions_of


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DQNTrainResult(TrainResult):
    """Training summary for :class:`DQNAgent`."""

    obs_dim: int = 0
    mean_loss: float = 0.0

    def __str__(self) -> str:
        return (
            f"DQNTrainResult  episodes={self.n_episodes}"
            f"  obs_dim={self.obs_dim}"
            f"  mean_reward={self.mean_reward:.4f}"
            f"  best_reward={self.best_reward:.4f}"
            f"  mean_loss={self.mean_loss:.6f}"
            f"  final_epsilon={self.final_epsilon:.4f}"
        )


# ── Tiny NumPy neural net ─────────────────────────────────────────────────────

class _MLP:
    """
    Two-hidden-layer fully-connected network.

    Parameters
    ----------
    in_dim, hidden, out_dim:
        Layer dimensions.
    lr:
        Learning rate for SGD weight updates.
    seed:
        NumPy random seed for weight initialisation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        lr: float = 1e-3,
        seed: Optional[int] = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        # Xavier / Glorot uniform init
        def _w(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float64)

        self.W1 = _w(in_dim, hidden)
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = _w(hidden, hidden)
        self.b2 = np.zeros(hidden, dtype=np.float64)
        self.W3 = _w(hidden, out_dim)
        self.b3 = np.zeros(out_dim, dtype=np.float64)
        self.lr = lr

    # ── Forward (single sample or batch) ─────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, in_dim) or (in_dim,)
        returns: (batch, out_dim) or (out_dim,)
        """
        self._x = x
        self._z1 = x @ self.W1 + self.b1
        self._a1 = np.maximum(0.0, self._z1)   # ReLU
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = np.maximum(0.0, self._z2)   # ReLU
        self._out = self._a2 @ self.W3 + self.b3
        return self._out

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Stateless forward pass (does not store activations)."""
        a1 = np.maximum(0.0, x @ self.W1 + self.b1)
        a2 = np.maximum(0.0, a1 @ self.W2 + self.b2)
        return a2 @ self.W3 + self.b3

    # ── Backward + SGD step ───────────────────────────────────────────────────

    def backward(self, loss_grad: np.ndarray) -> None:
        """
        loss_grad: ∂L/∂output, shape (batch, out_dim).
        Applies one SGD step to all weight matrices.
        """
        batch = self._x.shape[0]

        # Layer 3
        dW3 = self._a2.T @ loss_grad / batch
        db3 = loss_grad.mean(axis=0)
        d2  = (loss_grad @ self.W3.T) * (self._z2 > 0)

        # Layer 2
        dW2 = self._a1.T @ d2 / batch
        db2 = d2.mean(axis=0)
        d1  = (d2 @ self.W2.T) * (self._z1 > 0)

        # Layer 1
        dW1 = self._x.T @ d1 / batch
        db1 = d1.mean(axis=0)

        # SGD update
        self.W3 -= self.lr * dW3;  self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1

    # ── Weight copy ───────────────────────────────────────────────────────────

    def copy_weights_from(self, other: "_MLP") -> None:
        """Hard-copy all weights from *other* into self (target-network update)."""
        self.W1[:] = other.W1;  self.b1[:] = other.b1
        self.W2[:] = other.W2;  self.b2[:] = other.b2
        self.W3[:] = other.W3;  self.b3[:] = other.b3

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k).tolist() for k in ("W1","b1","W2","b2","W3","b3")}

    def from_dict(self, d: dict[str, Any]) -> None:
        for k in ("W1","b1","W2","b2","W3","b3"):
            setattr(self, k, np.array(d[k], dtype=np.float64))


# ── Replay buffer ─────────────────────────────────────────────────────────────

class _ReplayBuffer:
    """Circular experience-replay buffer."""

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self._buf: deque[tuple] = deque(maxlen=capacity)
        self._rng = rng

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        idxs = self._rng.integers(0, len(self._buf), size=batch_size)
        batch = [self._buf[i] for i in idxs]
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float64),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float64),
            np.array(next_obs, dtype=np.float64),
            np.array(dones,    dtype=np.float64),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ── Agent ─────────────────────────────────────────────────────────────────────

class DQNAgent(AgentBase):
    """
    Deep Q-Network agent (pure NumPy, no framework required).

    Parameters
    ----------
    n_actions:
        Discrete action-space size.  Pass 0 to auto-detect from the
        environment on the first ``train()`` call.
    obs_dim:
        Observation vector dimension.  Pass 0 to infer from the first
        observed state.  Observations are flattened before being fed to
        the network (strings are hashed to a single float).
    hidden_size:
        Number of units in each hidden layer.
    lr:
        SGD learning rate.
    gamma:
        Discount factor.
    epsilon:
        Initial exploration rate.
    epsilon_min:
        Minimum exploration rate after decay.
    epsilon_decay:
        Per-episode multiplicative ε decay.
    buffer_size:
        Replay buffer capacity.
    batch_size:
        Mini-batch size drawn from the replay buffer each step.
    target_update_freq:
        Number of *steps* between hard target-network copies.
    seed:
        Random seed for weight initialisation and replay sampling.
    """

    name = "dqn"

    def __init__(
        self,
        n_actions: int = 0,
        obs_dim: int = 0,
        *,
        hidden_size: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._seed = seed
        self._rng_py = random.Random(seed)
        self._rng_np = np.random.default_rng(seed)

        # Built lazily on first obs
        self._online: Optional[_MLP] = None
        self._target: Optional[_MLP] = None
        self._buffer: Optional[_ReplayBuffer] = None
        self._global_step = 0

    # ── Lazy init ─────────────────────────────────────────────────────────────

    def _init_nets(self, obs_dim: int) -> None:
        if self._online is not None:
            return
        self.obs_dim = obs_dim
        net_seed = self._seed
        self._online = _MLP(obs_dim, self.hidden_size, self.n_actions, self.lr, net_seed)
        self._target = _MLP(obs_dim, self.hidden_size, self.n_actions, self.lr, net_seed)
        self._target.copy_weights_from(self._online)
        self._buffer = _ReplayBuffer(self.buffer_size, self._rng_np)

    # ── Public API ────────────────────────────────────────────────────────────

    def act(self, obs: Any) -> int:
        """Greedy action selection (no exploration)."""
        if self._online is None:
            return 0  # not yet trained
        x = np.array(_flat_obs(obs), dtype=np.float64)
        q_vals = self._online.predict(x)
        return int(np.argmax(q_vals))

    def train(
        self,
        env: Any,
        n_episodes: int = 300,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ) -> DQNTrainResult:
        """
        Train DQN on *env* for *n_episodes* episodes.

        Parameters
        ----------
        env:
            Any RLIP environment with a discrete action space.
        n_episodes:
            Number of training episodes.
        max_steps:
            Step cap per episode.
        seed:
            Base environment seed.

        Returns
        -------
        DQNTrainResult
        """
        if self.n_actions == 0:
            self.n_actions = _n_actions_of(env)

        episode_rewards: list[float] = []
        total_losses: list[float] = []
        best_ep_reward  = float("-inf")
        best_ep_history: list[tuple] = []

        for ep in range(n_episodes):
            ep_seed = (seed + ep) if seed is not None else None
            reset_out = env.reset(seed=ep_seed)
            obs = _get(reset_out, "observation", reset_out)
            obs_vec = np.array(_flat_obs(obs), dtype=np.float64)

            # Build networks on first observation
            self._init_nets(obs_vec.shape[0])

            total_reward = 0.0
            ep_history: list[tuple] = []

            for _ in range(max_steps):
                # ε-greedy action
                if self._rng_py.random() < self.epsilon:
                    action = self._rng_py.randint(0, self.n_actions - 1)
                else:
                    q_vals = self._online.predict(obs_vec)  # type: ignore[union-attr]
                    action = int(np.argmax(q_vals))

                ep_history.append((obs, action))
                step_out = env.step(action)
                next_obs  = _get(step_out, "observation", obs)
                reward     = float(_get(step_out, "reward", 0.0))
                terminated = bool(_get(step_out, "terminated", False))
                truncated  = bool(_get(step_out, "truncated", False))
                done = terminated or truncated

                next_obs_vec = np.array(_flat_obs(next_obs), dtype=np.float64)
                self._buffer.push(obs_vec, action, reward, next_obs_vec, done)  # type: ignore[union-attr]
                total_reward += reward
                obs_vec = next_obs_vec
                obs = next_obs
                self._global_step += 1

                # Learn
                if len(self._buffer) >= self.batch_size:  # type: ignore[arg-type]
                    loss = self._learn()
                    total_losses.append(loss)

                # Target network update
                if self._global_step % self.target_update_freq == 0:
                    self._target.copy_weights_from(self._online)  # type: ignore[union-attr]

                if done:
                    break

            episode_rewards.append(total_reward)
            if total_reward > best_ep_reward:
                best_ep_reward  = total_reward
                best_ep_history = list(ep_history)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        mean_loss = float(np.mean(total_losses)) if total_losses else 0.0
        return DQNTrainResult(
            agent_name=self.name,
            n_episodes=n_episodes,
            episode_rewards=episode_rewards,
            final_epsilon=self.epsilon,
            best_episode_history=best_ep_history,
            obs_dim=self.obs_dim,
            mean_loss=mean_loss,
        )

    def save(self, path: str | Path) -> None:
        """Save network weights and hyper-parameters to a JSON file."""
        data: dict[str, Any] = {
            "agent": self.name,
            "n_actions": self.n_actions,
            "obs_dim": self.obs_dim,
            "hidden_size": self.hidden_size,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
        }
        if self._online is not None:
            data["online_weights"] = self._online.to_dict()
            data["target_weights"] = self._target.to_dict()  # type: ignore[union-attr]
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Restore network weights and hyper-parameters from a JSON file."""
        data = json.loads(Path(path).read_text())
        self.n_actions        = data["n_actions"]
        self.obs_dim          = data["obs_dim"]
        self.hidden_size      = data["hidden_size"]
        self.lr               = data["lr"]
        self.gamma            = data["gamma"]
        self.epsilon          = data["epsilon"]
        self.epsilon_min      = data["epsilon_min"]
        self.epsilon_decay    = data["epsilon_decay"]
        self.buffer_size      = data["buffer_size"]
        self.batch_size       = data["batch_size"]
        self.target_update_freq = data["target_update_freq"]
        if "online_weights" in data:
            self._init_nets(self.obs_dim)
            self._online.from_dict(data["online_weights"])   # type: ignore[union-attr]
            self._target.from_dict(data["target_weights"])   # type: ignore[union-attr]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _learn(self) -> float:
        """Draw a mini-batch from the replay buffer and update the online net."""
        obs_b, act_b, rew_b, next_b, done_b = self._buffer.sample(self.batch_size)  # type: ignore[union-attr]

        # Target Q-values from target network
        q_next = self._target.predict(next_b)                 # type: ignore[union-attr]
        q_target_full = self._online.forward(obs_b).copy()    # type: ignore[union-attr]

        # Bellman targets
        best_next = q_next.max(axis=1)
        batch_idx = np.arange(self.batch_size)
        q_target_full[batch_idx, act_b] = (
            rew_b + self.gamma * best_next * (1.0 - done_b)
        )

        # Compute loss gradient: dL/dout = 2*(Q_pred - target) / batch
        q_pred = self._online._out                             # already computed in forward
        loss_grad = 2.0 * (q_pred - q_target_full) / self.batch_size
        self._online.backward(loss_grad)                       # type: ignore[union-attr]

        loss = float(np.mean((q_pred - q_target_full) ** 2))
        return loss
