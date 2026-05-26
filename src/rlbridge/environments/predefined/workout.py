"""
rlbridge Simulated Workout Environment
=====================================
Discrete toy environment for training scheduling and fatigue management.

State
-----
The hidden state tracks:
    day:          training day index (0..max_days)
    fitness:      adaptation score (0..20)
    fatigue:      recovery debt (0..10)
    rest_streak:  consecutive rest days (0..max_days)

Actions
-------
    0 - rest
    1 - light workout
    2 - moderate workout
    3 - hard workout

Objective
---------
Grow fitness while controlling fatigue. The episode terminates successfully
when fitness reaches the target and fatigue is still manageable.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from ...protocol.messages import (
    DiscreteSpace,
    EnvironmentInfo,
    MultiDiscreteSpace,
    RenderResult,
    ResetResult,
    StepResult,
    SuggestedHyperparameters,
)
from ..base import rlbridgeEnvironment, rlbridgeEnvironmentFactory


class WorkoutEnvironment(rlbridgeEnvironment):
    """Simple deterministic workout-planning simulator."""

    def __init__(
        self,
        *,
        max_days: int = 30,
        target_fitness: int = 18,
    ) -> None:
        self._max_days = max_days
        self._target_fitness = target_fitness

        self._day = 0
        self._fitness = 6
        self._fatigue = 2
        self._rest_streak = 0
        self._initialized = False
        self._rng = random.Random()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        if seed is not None:
            self._rng.seed(seed)

        opts = options or {}
        self._day = int(opts.get("start_day", 0))
        self._fitness = int(opts.get("start_fitness", 6))
        self._fatigue = int(opts.get("start_fatigue", 2))
        self._rest_streak = int(opts.get("start_rest_streak", 0))

        self._fitness = max(0, min(20, self._fitness))
        self._fatigue = max(0, min(10, self._fatigue))
        self._rest_streak = max(0, min(self._max_days, self._rest_streak))
        self._day = max(0, min(self._max_days, self._day))

        self._initialized = True
        return ResetResult(observation=self._obs(), info=self._info())

    def step(self, action: Any) -> StepResult:
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")

        a = int(action)
        if a not in (0, 1, 2, 3):
            raise ValueError(f"Action must be 0..3, got {a}")

        prev_fatigue = self._fatigue
        load = [0, 1, 2, 3][a]
        recovery = [2, 1, 1, 0][a]
        self._fatigue = max(0, min(10, prev_fatigue + load - recovery))

        if a == 0:
            self._rest_streak += 1
        else:
            self._rest_streak = 0

        gain = 0
        if a == 0:
            gain = -1 if (self._rest_streak >= 3 and self._fitness > 0) else 0
        elif a == 1:
            gain = 1 if prev_fatigue <= 6 else 0
        elif a == 2:
            gain = 2 if prev_fatigue <= 5 else -1
        elif a == 3:
            if prev_fatigue <= 3:
                gain = 3
            elif prev_fatigue >= 8:
                gain = -2
            else:
                gain = 0

        if self._fatigue >= 9:
            gain -= 1

        self._fitness = max(0, min(20, self._fitness + gain))
        self._day += 1

        reward = gain * 0.5
        reward -= 0.2 * max(0, self._fatigue - 7)
        if a == 0 and prev_fatigue >= 6:
            reward += 0.3
        if a in (2, 3) and self._fatigue <= 6:
            reward += 0.1

        terminated = self._fitness >= self._target_fitness and self._fatigue <= 6
        truncated = self._day >= self._max_days

        if terminated or truncated:
            self._initialized = False

        return StepResult(
            observation=self._obs(),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=self._info(),
        )

    def close(self) -> None:
        self._initialized = False

    @property
    def observation_space(self) -> MultiDiscreteSpace:
        return MultiDiscreteSpace(nvec=[self._max_days + 1, 21, 11, self._max_days + 1])

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(n=4, start=0)

    def render(self) -> RenderResult:
        action_guide = "0=rest 1=light 2=moderate 3=hard"
        text = (
            f"day={self._day}/{self._max_days} "
            f"fitness={self._fitness} "
            f"fatigue={self._fatigue} "
            f"rest_streak={self._rest_streak}  "
            f"actions[{action_guide}]"
        )
        return RenderResult(mode="ansi", text=text)

    def sample_action(self) -> int:
        return self._rng.randint(0, 3)

    def _obs(self) -> list[int]:
        return [self._day, self._fitness, self._fatigue, self._rest_streak]

    def _info(self) -> dict[str, Any]:
        return {
            "day": self._day,
            "fitness": self._fitness,
            "fatigue": self._fatigue,
            "rest_streak": self._rest_streak,
            "target_fitness": self._target_fitness,
            "max_days": self._max_days,
        }


class WorkoutFactory(rlbridgeEnvironmentFactory):
    """Factory for predefined workout simulators."""

    def __init__(
        self,
        env_id: str = "Workout-Sim-v0",
        *,
        max_days: int = 30,
        target_fitness: int = 18,
        tags: Optional[list[str]] = None,
        description: str = "",
    ) -> None:
        self._env_id = env_id
        self._max_days = max_days
        self._target_fitness = target_fitness
        self._tags = tags or ["workout", "health", "planning", "discrete", "simulated"]
        self._description = description or (
            "Workout planning simulator with fatigue management. Choose daily "
            "training intensity to reach target fitness before time runs out."
        )

    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            env_id=self._env_id,
            description=self._description,
            tags=self._tags,
            namespace="workout",
            render_modes=["ansi"],
            max_episode_steps=self._max_days,
            reward_threshold=5.0,
            suggested_hyperparameters=SuggestedHyperparameters(
                agent_type="ppo",
                n_episodes=1200,
                max_steps=self._max_days,
                gamma=0.98,
                hidden_size=128,
                lr=3e-4,
                sub_goal_threshold=0.5,
                top_k=3,
                min_episode_visits=2,
            ),
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> WorkoutEnvironment:
        _ = render_mode
        return WorkoutEnvironment(
            max_days=int(kwargs.get("max_days", self._max_days)),
            target_fitness=int(kwargs.get("target_fitness", self._target_fitness)),
        )


WORKOUT_SIM_V0 = WorkoutFactory()
ALL_WORKOUT_FACTORIES: list[WorkoutFactory] = [WORKOUT_SIM_V0]
