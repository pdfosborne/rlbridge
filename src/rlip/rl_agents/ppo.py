"""
Proximal Policy Optimisation (PPO) Agent
=========================================
A from-scratch NumPy implementation of PPO-Clip for discrete action spaces.

Network architecture
--------------------
Actor and critic share no weights.  Both are two-hidden-layer MLPs::

    obs_dim → hidden_size → hidden_size → n_actions   (actor, softmax output)
    obs_dim → hidden_size → hidden_size → 1            (critic, scalar value)

Algorithm  (PPO-Clip, Schulman et al. 2017)
-------------------------------------------
For each iteration:

1. Collect ``n_steps`` transitions following the current policy π_θ_old.
   At each step:
   - Sample action from softmax(actor(obs)).
   - Store ``(obs, action, reward, done, log_prob_old, value_old)``.

2. Compute returns and advantages using Generalised Advantage Estimation
   (GAE-λ)::

       δ_t  = r_t + γ · V(s_{t+1}) · (1−done) − V(s_t)
       A_t  = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}
       G_t  = A_t + V(s_t)   (return / value target)

3. For ``ppo_epochs`` passes over the collected data in random mini-batches:
   a. Recompute log-probabilities log π_θ(a|s) and value estimates V_θ(s).
   b. Probability ratio: ``r = exp(log π_θ − log π_θ_old)``
   c. Clipped actor loss::

          L_clip = −E[min(r·A, clip(r, 1−ε, 1+ε)·A)]

   d. Critic loss: ``L_vf = MSE(V_θ(s), G)``
   e. Entropy bonus: ``−H[π_θ(·|s)]`` (encourages exploration).
   f. Total loss: ``L_clip + c_vf · L_vf − c_ent · H``

4. Update actor and critic with one SGD step per mini-batch.

References
----------
Schulman et al. (2017), "Proximal Policy Optimization Algorithms",
arXiv:1707.06347.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import copy

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    raise ImportError("PPO agent requires PyTorch. Install with: pip install torch")

from .._agent_base import AgentBase, TrainResult, _flat_obs, _get, _n_actions_of

# Use GPU when available; falls back to CPU transparently.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PPOTrainResult(TrainResult):
    """Training summary for :class:`PPOAgent`."""

    obs_dim: int = 0
    mean_actor_loss: float = 0.0
    mean_critic_loss: float = 0.0
    n_iterations: int = 0

    def __str__(self) -> str:
        return (
            f"PPOTrainResult  episodes={self.n_episodes}"
            f"  obs_dim={self.obs_dim}"
            f"  mean_reward={self.mean_reward:.4f}"
            f"  best_reward={self.best_reward:.4f}"
            f"  mean_actor_loss={self.mean_actor_loss:.6f}"
            f"  mean_critic_loss={self.mean_critic_loss:.6f}"
        )


# ── PyTorch MLP (GPU-accelerated when available; shared structure for actor/critic) ──

class _MLP:
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        lr: float,
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(x, dtype=torch.float64, device=_DEVICE)
        self._out_t = self._net(t)
        self._out = self._out_t.detach().cpu().numpy()
        return self._out

    def predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float64, device=_DEVICE)
            return self._net(t).cpu().numpy()

    def backward(self, loss_grad: np.ndarray) -> None:
        grad_t = torch.as_tensor(loss_grad, dtype=torch.float64, device=_DEVICE)
        self._opt.zero_grad()
        assert self._out_t is not None
        self._out_t.backward(grad_t)
        self._opt.step()

    def to_dict(self) -> dict[str, Any]:
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(x), x, 0.0)  # replace NaN/inf before softmax
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    s = e.sum(axis=-1, keepdims=True)
    # If sum is zero (all -inf inputs), return uniform
    s = np.where(s == 0, 1.0, s)
    return e / s


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(x), x, 0.0)  # replace NaN/inf before log-softmax
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def _gae(
    rewards: np.ndarray,     # (T,)
    values: np.ndarray,      # (T,)
    next_values: np.ndarray, # (T,)
    dones: np.ndarray,       # (T,)
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generalised Advantage Estimation.

    Returns advantages and returns (value targets), both shape (T,).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ── Agent ─────────────────────────────────────────────────────────────────────

class PPOAgent(AgentBase):
    """
    Proximal Policy Optimisation agent (pure NumPy, no framework required).

    Parameters
    ----------
    n_actions:
        Discrete action-space size.  Pass 0 to auto-detect from the
        environment on the first ``train()`` call.
    obs_dim:
        Flat observation dimension.  Pass 0 to infer from the first step.
    hidden_size:
        Hidden-layer width for both actor and critic networks.
    lr_actor:
        SGD learning rate for the actor network.
    lr_critic:
        SGD learning rate for the critic network.
    gamma:
        Discount factor for returns.
    lam:
        GAE λ parameter (0 = TD(0), 1 = Monte-Carlo).
    clip_eps:
        PPO probability-ratio clipping coefficient (ε).
    c_vf:
        Weighting coefficient for the critic (value) loss.
    c_ent:
        Entropy bonus coefficient (encourages exploration).
    n_steps:
        Number of environment steps collected per PPO iteration.
    ppo_epochs:
        Number of optimisation passes over each collected batch.
    mini_batch_size:
        Mini-batch size for each gradient update within an epoch.
    seed:
        Random seed for weight initialisation and action sampling.
    """

    name = "ppo"

    def __init__(
        self,
        n_actions: int = 0,
        obs_dim: int = 0,
        *,
        hidden_size: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        c_vf: float = 0.5,
        c_ent: float = 0.01,
        n_steps: int = 256,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        seed: Optional[int] = None,
    ) -> None:
        self.n_actions      = n_actions
        self.obs_dim        = obs_dim
        self.hidden_size    = hidden_size
        self.lr_actor       = lr_actor
        self.lr_critic      = lr_critic
        self.gamma          = gamma
        self.lam            = lam
        self.clip_eps       = clip_eps
        self.c_vf           = c_vf
        self.c_ent          = c_ent
        self.n_steps        = n_steps
        self.ppo_epochs     = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self._seed          = seed
        self._rng_py        = random.Random(seed)
        self._rng_np        = np.random.default_rng(seed)

        self._actor:  Optional[_MLP] = None
        self._critic: Optional[_MLP] = None

    # ── Lazy init ─────────────────────────────────────────────────────────────

    def _init_nets(self, obs_dim: int) -> None:
        if self._actor is not None:
            return
        self.obs_dim = obs_dim
        s = self._seed
        self._actor  = _MLP(obs_dim, self.hidden_size, self.n_actions, self.lr_actor,  s)
        self._critic = _MLP(obs_dim, self.hidden_size, 1,              self.lr_critic, s)

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
        """Sample an action from the current policy (stochastic)."""
        if self._actor is None:
            return 0
        x = self._obs_to_vec(obs)
        logits = self._actor.predict(x)
        probs = _softmax(logits)
        return int(self._rng_np.choice(self.n_actions, p=probs))

    def act_greedy(self, obs: Any) -> int:
        """Return the most probable action (greedy / deterministic)."""
        if self._actor is None:
            return 0
        x = self._obs_to_vec(obs)
        logits = self._actor.predict(x)
        return int(np.argmax(logits))

    def train(
        self,
        env: Any,
        n_episodes: int = 200,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ) -> PPOTrainResult:
        """
        Train PPO on *env*.

        Training is structured in iterations; each iteration collects
        ``n_steps`` environment transitions and then runs ``ppo_epochs``
        passes of mini-batch gradient updates.  Episodes are tracked for
        reporting.

        Parameters
        ----------
        env:
            Any RLIP environment with a discrete action space.
        n_episodes:
            Approximate total number of episodes (used to set total steps
            as ``n_episodes × max_steps``; actual count may differ slightly
            since episodes are allowed to run to completion).
        max_steps:
            Hard per-episode step cap when collecting rollouts.
        seed:
            Base environment seed.

        Returns
        -------
        PPOTrainResult
        """
        if self.n_actions == 0:
            self.n_actions = _n_actions_of(env)

        total_steps = n_episodes * max_steps
        episode_rewards: list[float] = []
        actor_losses:  list[float] = []
        critic_losses: list[float] = []
        completed_episodes = 0
        best_ep_reward  = float("-inf")
        best_ep_history: list[tuple] = []

        # Running episode state
        ep_seed = seed
        reset_out = env.reset(seed=ep_seed)
        obs = _get(reset_out, "observation", reset_out)
        obs_vec = np.array(_flat_obs(obs), dtype=np.float64)
        self._init_nets(obs_vec.shape[0])
        obs_vec = self._obs_to_vec(obs)
        current_ep_reward = 0.0
        current_ep_steps  = 0
        current_ep_history: list[tuple] = []
        global_step = 0

        while global_step < total_steps:
            # ── Rollout collection ─────────────────────────────────────────────
            rollout_obs:       list[np.ndarray] = []
            rollout_actions:   list[int]        = []
            rollout_rewards:   list[float]      = []
            rollout_dones:     list[float]      = []
            rollout_log_probs: list[float]      = []
            rollout_values:    list[float]      = []

            for _ in range(self.n_steps):
                logits = self._actor.forward(obs_vec[None, :])   # type: ignore[union-attr]
                log_probs_all = _log_softmax(logits)[0]
                probs = _softmax(logits)[0]
                action = int(self._rng_np.choice(self.n_actions, p=probs))
                log_prob = float(log_probs_all[action])

                value = float(self._critic.predict(obs_vec[None, :]).flatten()[0])  # type: ignore[union-attr]

                step_out = env.step(action)
                next_obs   = _get(step_out, "observation", obs)
                reward     = float(_get(step_out, "reward", 0.0))
                terminated = bool(_get(step_out, "terminated", False))
                truncated  = bool(_get(step_out, "truncated", False))
                done = terminated or truncated

                rollout_obs.append(obs_vec)
                rollout_actions.append(action)
                rollout_rewards.append(reward)
                rollout_dones.append(float(done))
                rollout_log_probs.append(log_prob)
                rollout_values.append(value)
                current_ep_history.append((obs, action))

                current_ep_reward += reward
                current_ep_steps  += 1
                global_step += 1

                if done or current_ep_steps >= max_steps:
                    episode_rewards.append(current_ep_reward)
                    if current_ep_reward > best_ep_reward:
                        best_ep_reward  = current_ep_reward
                        best_ep_history = list(current_ep_history)
                    completed_episodes += 1
                    current_ep_reward   = 0.0
                    current_ep_steps    = 0
                    current_ep_history  = []
                    ep_seed = (seed + completed_episodes) if seed is not None else None
                    reset_out = env.reset(seed=ep_seed)
                    obs = _get(reset_out, "observation", reset_out)
                    obs_vec = self._obs_to_vec(obs)
                else:
                    obs = next_obs
                    obs_vec = self._obs_to_vec(next_obs)

                if global_step >= total_steps:
                    break

            T = len(rollout_obs)
            obs_arr     = np.array(rollout_obs,       dtype=np.float64)    # (T, D)
            act_arr     = np.array(rollout_actions,   dtype=np.int64)      # (T,)
            values_arr  = np.array(rollout_values,    dtype=np.float64)    # (T,)
            log_old_arr = np.array(rollout_log_probs, dtype=np.float64)    # (T,)
            dones_arr   = np.array(rollout_dones,     dtype=np.float64)    # (T,)

            # Bootstrap value of state after last rollout step
            next_val = float(self._critic.predict(obs_vec[None, :]).flatten()[0])  # type: ignore[union-attr]
            next_vals_arr = np.append(values_arr[1:], next_val)

            advantages, returns = _gae(
                np.array(rollout_rewards, dtype=np.float64),
                values_arr,
                next_vals_arr,
                dones_arr,
                self.gamma,
                self.lam,
            )
            # Normalise advantages
            if T > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ── PPO update epochs ──────────────────────────────────────────────
            indices = np.arange(T)
            for _ in range(self.ppo_epochs):
                self._rng_np.shuffle(indices)
                for start in range(0, T, self.mini_batch_size):
                    mb_idx = indices[start: start + self.mini_batch_size]
                    if len(mb_idx) == 0:
                        continue

                    mb_obs  = obs_arr[mb_idx]                   # (B, D)
                    mb_acts = act_arr[mb_idx]                   # (B,)
                    mb_adv  = advantages[mb_idx]                # (B,)
                    mb_ret  = returns[mb_idx]                   # (B,)
                    mb_lp_old = log_old_arr[mb_idx]             # (B,)

                    # ── Actor forward ──────────────────────────────────────────
                    logits_new = self._actor.forward(mb_obs)    # type: ignore[union-attr]
                    log_probs_new = _log_softmax(logits_new)    # (B, A)
                    probs_new = _softmax(logits_new)            # (B, A)
                    B = mb_obs.shape[0]
                    lp_new = log_probs_new[np.arange(B), mb_acts]  # (B,)

                    # ── PPO-clip actor loss ────────────────────────────────────
                    ratio   = np.exp(lp_new - mb_lp_old)
                    clip_r  = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    actor_loss = -np.minimum(ratio * mb_adv, clip_r * mb_adv).mean()

                    # Entropy bonus: H = -Σ p·log(p)
                    entropy = -(probs_new * log_probs_new).sum(axis=1).mean()
                    actor_losses.append(float(actor_loss))

                    # Actor gradient: ∂L/∂logits
                    # dL/d(log_p_a) = -min(r, clip_r) * A / B
                    # Using chain rule through log-softmax:
                    clipped = (clip_r < ratio) | (clip_r > ratio)  # True where ratio was clipped
                    used_ratio = np.where(ratio <= clip_r, ratio, clip_r)
                    grad_logp  = -(used_ratio * mb_adv) / B        # (B,)
                    # Entropy grad: ∂(-H)/∂p_i = log(p_i) + 1; via softmax chain rule
                    # Full grad wrt logits (through log-softmax and then -entropy)
                    grad_logits = np.zeros_like(logits_new)
                    grad_logits[np.arange(B), mb_acts] += grad_logp
                    # Entropy gradient wrt logits: -(∂H/∂logits) = probs*(log_probs+1) - mean
                    ent_grad = probs_new * (log_probs_new + 1.0) - \
                               (probs_new * (log_probs_new + 1.0)).sum(axis=1, keepdims=True)
                    grad_logits += self.c_ent * ent_grad

                    self._actor.backward(grad_logits)             # type: ignore[union-attr]

                    # ── Critic update ──────────────────────────────────────────
                    val_pred = self._critic.forward(mb_obs).flatten()   # type: ignore[union-attr]
                    crit_loss = float(np.mean((val_pred - mb_ret) ** 2))
                    critic_losses.append(crit_loss)

                    grad_val = 2.0 * (val_pred - mb_ret) / B
                    self._critic.backward(self.c_vf * grad_val[:, None])   # type: ignore[union-attr]

        return PPOTrainResult(
            agent_name=self.name,
            n_episodes=completed_episodes,
            episode_rewards=episode_rewards,
            final_epsilon=0.0,   # PPO uses stochastic policy, no ε
            best_episode_history=best_ep_history,
            obs_dim=self.obs_dim,
            mean_actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.0,
            mean_critic_loss=float(np.mean(critic_losses)) if critic_losses else 0.0,
            n_iterations=global_step // self.n_steps,
        )

    def save(self, path: str | Path) -> None:
        """Save actor/critic weights and hyper-parameters to a JSON file."""
        data: dict[str, Any] = {
            "agent": self.name,
            "n_actions":      self.n_actions,
            "obs_dim":        self.obs_dim,
            "hidden_size":    self.hidden_size,
            "lr_actor":       self.lr_actor,
            "lr_critic":      self.lr_critic,
            "gamma":          self.gamma,
            "lam":            self.lam,
            "clip_eps":       self.clip_eps,
            "c_vf":           self.c_vf,
            "c_ent":          self.c_ent,
            "n_steps":        self.n_steps,
            "ppo_epochs":     self.ppo_epochs,
            "mini_batch_size": self.mini_batch_size,
        }
        if self._actor is not None:
            data["actor_weights"]  = self._actor.to_dict()
            data["critic_weights"] = self._critic.to_dict()  # type: ignore[union-attr]
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Restore actor/critic weights and hyper-parameters from a JSON file."""
        data = json.loads(Path(path).read_text())
        self.n_actions       = data["n_actions"]
        self.obs_dim         = data["obs_dim"]
        self.hidden_size     = data["hidden_size"]
        self.lr_actor        = data["lr_actor"]
        self.lr_critic       = data["lr_critic"]
        self.gamma           = data["gamma"]
        self.lam             = data["lam"]
        self.clip_eps        = data["clip_eps"]
        self.c_vf            = data["c_vf"]
        self.c_ent           = data["c_ent"]
        self.n_steps         = data["n_steps"]
        self.ppo_epochs      = data["ppo_epochs"]
        self.mini_batch_size = data["mini_batch_size"]
        if "actor_weights" in data:
            self._init_nets(self.obs_dim)
            self._actor.from_dict(data["actor_weights"])    # type: ignore[union-attr]
            self._critic.from_dict(data["critic_weights"])  # type: ignore[union-attr]
