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

import copy

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    raise ImportError("DQN agent requires PyTorch. Install with: pip install torch")

from .._agent_base import AgentBase, TrainResult, _flat_obs, _get, _n_actions_of

# Use GPU when available; falls back to CPU transparently.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ── PyTorch neural net (GPU-accelerated when available) ──────────────────────

class _MLP:
    """
    Two-hidden-layer fully-connected network backed by PyTorch.

    Automatically runs on GPU when :data:`_DEVICE` is ``"cuda"``.
    The external API (forward / predict / backward / to_dict / from_dict /
    copy_weights_from) is identical to the previous NumPy implementation so
    the rest of the agent code requires no changes.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        lr: float = 1e-3,
        seed: Optional[int] = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self._net = nn.Sequential(
            nn.Linear(in_dim, hidden, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden, out_dim, dtype=torch.float64),
        ).to(_DEVICE)
        for m in self._net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.lr = lr
        self._opt = torch.optim.SGD(self._net.parameters(), lr=lr)
        self._out_t: Optional[torch.Tensor] = None
        self._out: Optional[np.ndarray] = None

    # ── Forward (single sample or batch) ─────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, in_dim) or (in_dim,)
        returns: (batch, out_dim) or (out_dim,)
        Retains the computation graph for a subsequent :meth:`backward` call.
        """
        t = torch.as_tensor(x, dtype=torch.float64, device=_DEVICE)
        self._out_t = self._net(t)
        self._out = self._out_t.detach().cpu().numpy()
        return self._out

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Stateless forward pass - no gradient graph retained."""
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float64, device=_DEVICE)
            return self._net(t).cpu().numpy()

    # ── Backward + SGD step ───────────────────────────────────────────────────

    def backward(self, loss_grad: np.ndarray) -> None:
        """
        loss_grad: ∂L/∂output, shape (batch, out_dim).
        Applies one SGD step using the graph retained by the last forward().
        """
        grad_t = torch.as_tensor(loss_grad, dtype=torch.float64, device=_DEVICE)
        self._opt.zero_grad()
        assert self._out_t is not None
        self._out_t.backward(grad_t)
        self._opt.step()

    # ── Weight copy ───────────────────────────────────────────────────────────

    def copy_weights_from(self, other: "_MLP") -> None:
        """Hard-copy all weights from *other* into self (target-network update)."""
        self._net.load_state_dict(copy.deepcopy(other._net.state_dict()))

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise weights in the same format as the original NumPy version."""
        sd = self._net.state_dict()
        _keys = [
            ("0.weight", "W1"), ("0.bias", "b1"),
            ("2.weight", "W2"), ("2.bias", "b2"),
            ("4.weight", "W3"), ("4.bias", "b3"),
        ]
        result: dict[str, Any] = {}
        for pt_k, np_k in _keys:
            arr = sd[pt_k].cpu().numpy()
            if pt_k.endswith(".weight"):
                arr = arr.T  # (out, in) → (in, out) for backward compatibility
            result[np_k] = arr.tolist()
        return result

    def from_dict(self, d: dict[str, Any]) -> None:
        """Restore weights from a dict produced by :meth:`to_dict`."""
        _keys = [
            ("W1", "0.weight"), ("b1", "0.bias"),
            ("W2", "2.weight"), ("b2", "2.bias"),
            ("W3", "4.weight"), ("b3", "4.bias"),
        ]
        sd: dict[str, torch.Tensor] = {}
        for np_k, pt_k in _keys:
            arr = np.array(d[np_k], dtype=np.float64)
            if pt_k.endswith(".weight"):
                arr = arr.T  # (in, out) → (out, in)
            sd[pt_k] = torch.tensor(arr, dtype=torch.float64, device=_DEVICE)
        self._net.load_state_dict(sd)


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

    def _obs_to_vec(self, obs: Any) -> np.ndarray:
        """Flatten and coerce *obs* to a fixed-size vector.

        If obs dimensionality changes across steps (common for dict/list states),
        vectors are padded/truncated to the first observed dimension.
        """
        vec = np.array(_flat_obs(obs), dtype=np.float64)
        vec = np.where(np.isfinite(vec), vec, 0.0)  # sanitize NaN/inf
        if self.obs_dim <= 0:
            return vec
        if vec.shape[0] == self.obs_dim:
            return vec
        if vec.shape[0] < self.obs_dim:
            pad = np.zeros(self.obs_dim - vec.shape[0], dtype=np.float64)
            return np.concatenate([vec, pad])
        return vec[: self.obs_dim]

    # ── Public API ────────────────────────────────────────────────────────────

    def act(self, obs: Any) -> int:
        """Greedy action selection (no exploration)."""
        if self._online is None:
            return 0  # not yet trained
        x = self._obs_to_vec(obs)
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
            Any rlbridge environment with a discrete action space.
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
            obs_vec = self._obs_to_vec(obs)

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

                next_obs_vec = self._obs_to_vec(next_obs)
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
