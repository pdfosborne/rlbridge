"""
RLIP 1-D Grid World Environment
================================
Simple discrete grid world used as a bundled example environment.

Physics / Rules
---------------
A 1-D grid of SIZE cells (default 10).  The agent starts in the left half and
must navigate right to reach the goal cell (position SIZE-1).

Actions
-------
    0 – move left  (clamps at 0)
    1 – move right (clamps at SIZE-1)

Observation
-----------
A list with a single integer: ``[position]``  (0 … SIZE-1).

Reward / Termination
--------------------
    +1.0  when the agent reaches position SIZE-1  (terminated)
    −0.01 every other step

Render
------
``ansi`` mode: ASCII bar showing the agent (A), empty cells (.), and goal (G/★).

Variants
--------
    ``GridWorld-1D-v0``  — default 10-cell grid, max 100 steps
"""

from __future__ import annotations

import random
from typing import Any, Optional

from ..base import RLIPEnvironment, RLIPEnvironmentFactory
from ...protocol.messages import (
    DiscreteSpace,
    EnvironmentInfo,
    RenderResult,
    ResetResult,
    StepResult,
)

class GridWorldEnv(RLIPEnvironment):
    """Simple 1-D Grid World environment."""

    SIZE = 10  # positions 0..9

    def __init__(self) -> None:
        self._pos: int = 0
        self._initialized = False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        rng = random.Random(seed)
        self._pos = rng.randint(0, self.SIZE // 2)  # start in left half
        self._initialized = True
        return ResetResult(observation=[self._pos], info={"size": self.SIZE})

    def step(self, action: Any) -> StepResult:
        if not self._initialized:
            raise RuntimeError("Call reset() first")
        a = int(action)
        if a == 0:
            self._pos = max(0, self._pos - 1)
        elif a == 1:
            self._pos = min(self.SIZE - 1, self._pos + 1)

        terminated = self._pos == self.SIZE - 1
        reward = 1.0 if terminated else -0.01
        if terminated:
            self._initialized = False

        return StepResult(
            observation=[self._pos],
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={"position": self._pos},
        )

    def close(self) -> None:
        self._initialized = False

    @property
    def observation_space(self) -> DiscreteSpace:
        return DiscreteSpace(n=self.SIZE, start=0)

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(n=2)

    def render(self) -> RenderResult:
        grid = ["." if i != self._pos else "A" for i in range(self.SIZE)]
        grid[-1] = "G"
        if self._pos == self.SIZE - 1:
            grid[-1] = "★"
        text = "".join(grid) + f"  pos={self._pos}"
        return RenderResult(mode="ansi", text=text)

    def sample_action(self) -> Any:
        return random.randint(0, 1)


class GridWorldFactory(RLIPEnvironmentFactory):
    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            env_id="GridWorld-1D-v0",
            description=(
                "Simple 1-D grid world: navigate right to reach the goal cell. "
                "Discrete(2) actions (left/right), integer position observation."
            ),
            tags=["custom", "simple", "discrete", "example"],
            max_episode_steps=100,
            reward_threshold=0.9,
            namespace="custom",
            render_modes=["ansi"],
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> GridWorldEnv:
        return GridWorldEnv()


GRIDWORLD_V0 = GridWorldFactory()
ALL_GRIDWORLD_FACTORIES: list[GridWorldFactory] = [GRIDWORLD_V0]
