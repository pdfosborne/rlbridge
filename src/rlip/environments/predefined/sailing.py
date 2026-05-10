"""
RLIP Sailing Environment
=========================
Native implementation of the simple sailing simulator originally described in
https://github.com/pdfosborne/elsciRL-App-Sailing.

Physics
-------
A sailboat starts at position (x=0, y=0, angle=0).  The agent chooses one of
two discrete actions at each step:

    0 – turn slightly left  (angle -= 0.1 rad)
    1 – turn slightly right (angle += 0.1 rad)

The boat's velocity is a Gaussian function of the angle relative to the wind
(which blows straight down the y-axis from above):

    vel(θ) = 1 − exp(−θ² / θ_dead)     θ_dead = π/12

Position update:
    x  += round(vel(θ') · sin(θ'), obs_precision)
    y  += round(vel(θ') · cos(θ'), 4)
    θ  ← round(θ', 1)           where θ' = θ + a

Termination & reward:
    |x| > x_limit   →  reward = −1, terminated  (hit pier)
    y  > y_limit     →  reward = +1, terminated  (reached goal)
    y  < 0           →  reward = −1, terminated  (sailed backwards)
    |θ| > π/2        →  reward = −1, terminated  (sailed backwards)
    otherwise        →  reward =  0 (or small supervised reward if enabled)

Observation
-----------
A text string ``"{x:.{precision}f}_{angle:.1f}"`` — e.g. ``"0.0300_0.2"``.

Action space
------------
Discrete(2): 0 = turn left, 1 = turn right.

Variants
--------
Two pre-configured variants are registered at import time:

  ``Sailing-v0``   — y_limit=25, obs_precision=4, supervised_rewards=False
  ``Sailing-Hard-v0`` — y_limit=50, obs_precision=4, supervised_rewards=True,
                        tighter x_limit=8

Usage
-----
    from rlip.environments.registry import registry

    factory = registry.get("Sailing-v0")
    env = factory.create()
    result = env.reset(seed=0)
    print(result.observation)   # e.g. "0.0000_0.0"
"""

from __future__ import annotations

import base64
import math
import random
from io import BytesIO
from typing import Any, Optional

from ...protocol.messages import (
    DiscreteSpace,
    EnvironmentInfo,
    RenderResult,
    ResetResult,
    SpaceDescription,
    StepResult,
    TextSpace,
)
from ..base import RLIPEnvironment, RLIPEnvironmentFactory


# ── Physics helpers ───────────────────────────────────────────────────────────

_THETA_DEAD = math.pi / 12


def _vel(theta: float, theta_0: float = 0.0) -> float:
    """Sailing velocity as a function of angle relative to wind."""
    return 1.0 - math.exp(-((theta - theta_0) ** 2) / _THETA_DEAD)


# ── Environment ───────────────────────────────────────────────────────────────

class SailingEnvironment(RLIPEnvironment):
    """
    RLIP sailing environment.

    Parameters
    ----------
    y_limit:
        Goal y-position — reaching y > y_limit gives reward +1.
    x_limit:
        Pier position — |x| > x_limit gives reward −1.
    obs_precision:
        Decimal places used when formatting the x part of the observation.
    supervised_rewards:
        If *True*, a small heading reward (vel(angle)/10) is added each step
        to encourage upwind sailing.
    render_mode:
        ``"rgb_array"`` to return base64-encoded PNG renders; ``None`` for
        no rendering.
    """

    def __init__(
        self,
        *,
        y_limit: float = 25.0,
        x_limit: float = 10.0,
        obs_precision: int = 4,
        supervised_rewards: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        self._y_limit = y_limit
        self._x_limit = x_limit
        self._obs_precision = obs_precision
        self._supervised_rewards = supervised_rewards
        self._render_mode = render_mode

        # Internal state — initialised by reset()
        self._x: float = 0.0
        self._y: float = 0.0
        self._angle: float = 0.0
        self._initialized: bool = False
        self._rng = random.Random()

    # ── Life-cycle ────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        if seed is not None:
            self._rng.seed(seed)

        start_obs: Optional[str] = (options or {}).get("start_obs")  # type: ignore[assignment]
        if start_obs:
            parts = start_obs.split("_")
            self._x = round(float(parts[0]), self._obs_precision)
            self._angle = round(float(parts[1]), 1)
        else:
            self._x = 0.0
            self._angle = 0.0
        self._y = 0.0
        self._initialized = True

        return ResetResult(
            observation=self._obs(),
            info={"x": self._x, "y": self._y, "angle": self._angle},
        )

    def step(self, action: Any) -> StepResult:
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")

        action = int(action)
        if action not in (0, 1):
            raise ValueError(f"Action must be 0 or 1, got {action}.")

        a = [-0.1, 0.1][action]
        new_angle = self._angle + a

        v = _vel(new_angle)
        self._x = round(self._x + v * math.sin(new_angle), self._obs_precision)
        self._y = round(self._y + v * math.cos(new_angle), 4)
        self._angle = round(new_angle, 1)

        # Reward and termination
        if abs(self._x) > self._x_limit:
            reward, terminated = -1.0, True
        elif self._y > self._y_limit:
            reward, terminated = 1.0, True
        elif self._y < 0.0:
            reward, terminated = -1.0, True
        elif abs(self._angle) > math.pi / 2:
            reward, terminated = -1.0, True
        else:
            reward = _vel(self._angle) / 10.0 if self._supervised_rewards else 0.0
            terminated = False

        if terminated:
            self._initialized = False

        return StepResult(
            observation=self._obs(),
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={"x": self._x, "y": self._y, "angle": self._angle},
        )

    def close(self) -> None:
        self._initialized = False

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def observation_space(self) -> SpaceDescription:
        return TextSpace()

    @property
    def action_space(self) -> SpaceDescription:
        return DiscreteSpace(n=2, start=0)

    def sample_action(self) -> int:
        return self._rng.randint(0, 1)

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self) -> RenderResult:
        if self._render_mode != "rgb_array":
            return RenderResult(mode=self._render_mode or "none")
        return self._render_rgb()

    def _render_rgb(self) -> RenderResult:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return RenderResult(mode="rgb_array", text="matplotlib not installed")

        x, y, angle = self._x, self._y, self._angle

        # Direction vector
        if angle < math.pi / 2:
            U, V = math.sin(angle), math.cos(angle)
        elif angle == math.pi / 2:
            U, V = 1.0, 0.0
        elif angle == -math.pi / 2:
            U, V = -1.0, 0.0
        else:
            U, V = math.sin(angle), -math.cos(angle)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=128)
        ax.scatter(x, y, c="b", marker="x", alpha=1)
        ax.quiver(x, y, U, V, angles="uv", scale_units="xy")
        if y > 1:
            ax.text(x + 0.5, y - 1, "Sailboat", color="b")

        # Wind indicator
        ax.quiver(0, self._y_limit, 0, -1, angles="uv", scale_units="xy", color="r")
        ax.text(0, self._y_limit + 0.25, "Wind", color="r")

        # Pier boundaries
        ax.plot([self._x_limit, self._x_limit], [0, self._y_limit], "r")
        ax.plot([-self._x_limit, -self._x_limit], [0, self._y_limit], "r")

        ax.set_title("Sailboat Position with Direction against Wind")
        ax.set_xlabel(f"Horizontal Position ({x})")
        ax.set_ylabel(f"Vertical Position ({y})")
        ax.set_xlim(-self._x_limit - 1, self._x_limit + 1)
        ax.set_ylim(-1, self._y_limit + 2)

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return RenderResult(mode="rgb_array", data=b64, width=640, height=640)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _obs(self) -> str:
        return f"{self._x:.{self._obs_precision}f}_{self._angle:.1f}"


# ── Factory ───────────────────────────────────────────────────────────────────

class SailingFactory(RLIPEnvironmentFactory):
    """
    Factory for :class:`SailingEnvironment` instances.

    Parameters
    ----------
    env_id:
        Registered environment ID, e.g. ``"Sailing-v0"``.
    y_limit, x_limit, obs_precision, supervised_rewards:
        Forwarded to :class:`SailingEnvironment`.
    tags:
        Metadata tags for the environment listing.
    max_episode_steps:
        Soft cap reported in environment metadata.
    """

    def __init__(
        self,
        env_id: str,
        *,
        y_limit: float = 25.0,
        x_limit: float = 10.0,
        obs_precision: int = 4,
        supervised_rewards: bool = False,
        tags: Optional[list[str]] = None,
        max_episode_steps: int = 200,
        description: str = "",
        reward_threshold: Optional[float] = None,
    ) -> None:
        self._env_id = env_id
        self._y_limit = y_limit
        self._x_limit = x_limit
        self._obs_precision = obs_precision
        self._supervised_rewards = supervised_rewards
        self._tags = tags or ["sailing", "navigation", "discrete"]
        self._max_episode_steps = max_episode_steps
        self._description = description or (
            f"Sailing simulator — steer a sailboat upwind to y>{y_limit:.0f} "
            f"within |x|<{x_limit:.0f}."
        )
        self._reward_threshold = reward_threshold

    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            env_id=self._env_id,
            description=self._description,
            tags=self._tags,
            namespace="sailing",
            render_modes=["rgb_array"],
            max_episode_steps=self._max_episode_steps,
            reward_threshold=self._reward_threshold,
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> SailingEnvironment:
        env = SailingEnvironment(
            y_limit=self._y_limit,
            x_limit=self._x_limit,
            obs_precision=self._obs_precision,
            supervised_rewards=self._supervised_rewards,
            render_mode=render_mode,
        )
        env.env_id = self._env_id  # type: ignore[attr-defined]
        return env


# ── Pre-configured variants ───────────────────────────────────────────────────

SAILING_V0 = SailingFactory(
    "Sailing-v0",
    y_limit=25.0,
    x_limit=10.0,
    obs_precision=4,
    supervised_rewards=False,
    tags=["sailing", "navigation", "discrete"],
    max_episode_steps=200,
    description=(
        "Simple sailing simulator: steer a sailboat upwind to y>25 "
        "within x-bounds of ±10.  Discrete action space (turn left / right)."
    ),
    reward_threshold=1.0,
)

SAILING_HARD_V0 = SailingFactory(
    "Sailing-Hard-v0",
    y_limit=50.0,
    x_limit=8.0,
    obs_precision=4,
    supervised_rewards=True,
    tags=["sailing", "navigation", "discrete", "hard"],
    max_episode_steps=500,
    description=(
        "Harder sailing variant: longer course (y>50), tighter x-bounds (±8), "
        "and supervised heading reward to guide exploration."
    ),
    reward_threshold=1.0,
)

#: All built-in sailing factories, in registration order.
ALL_SAILING_FACTORIES: list[SailingFactory] = [SAILING_V0, SAILING_HARD_V0]
