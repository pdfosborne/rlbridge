"""
rlbridge Simulated Meal Planning Environment
===========================================
Discrete meal-planning task with pantry and budget constraints.

State
-----
Observation is a list:
    [day, hunger, nutrition, budget_units, pantry]

Actions
-------
    0 - cook balanced meal (uses pantry, improves nutrition strongly)
    1 - cook quick meal    (uses pantry, cheaper gains)
    2 - grocery run        (spend budget to refill pantry)
    3 - skip meal          (no cost, worsens hunger/nutrition)

Goal
----
Maintain healthy nutrition while controlling hunger and budget over a weekly
planning horizon.
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


class MealPlanningEnvironment(rlbridgeEnvironment):
    """Simple meal-planning simulator for sequential decision making."""

    def __init__(
        self,
        *,
        max_days: int = 14,
        start_budget: int = 30,
    ) -> None:
        self._max_days = max_days
        self._start_budget = start_budget

        self._rng = random.Random()
        self._initialized = False

        self._day = 0
        self._hunger = 4
        self._nutrition = 5
        self._budget = start_budget
        self._pantry = 5

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        if seed is not None:
            self._rng.seed(seed)

        opts = options or {}
        self._day = int(opts.get("start_day", 0))
        self._hunger = int(opts.get("start_hunger", 4))
        self._nutrition = int(opts.get("start_nutrition", 5))
        self._budget = int(opts.get("start_budget", self._start_budget))
        self._pantry = int(opts.get("start_pantry", 5))

        self._day = self._clip(self._day, 0, self._max_days)
        self._hunger = self._clip(self._hunger, 0, 10)
        self._nutrition = self._clip(self._nutrition, 0, 10)
        self._budget = self._clip(self._budget, 0, 50)
        self._pantry = self._clip(self._pantry, 0, 10)

        self._initialized = True
        return ResetResult(observation=self._obs(), info=self._info())

    def step(self, action: Any) -> StepResult:
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")

        a = int(action)
        if a not in (0, 1, 2, 3):
            raise ValueError(f"Action must be 0..3, got {a}")

        invalid_penalty = 0.0

        # Baseline daily drift before action effects.
        self._hunger = self._clip(self._hunger + 1, 0, 10)
        self._nutrition = self._clip(self._nutrition - 1, 0, 10)

        if a == 0:  # balanced meal
            if self._pantry >= 2:
                self._pantry -= 2
                self._hunger = self._clip(self._hunger - 4, 0, 10)
                self._nutrition = self._clip(self._nutrition + 3, 0, 10)
            else:
                invalid_penalty = 0.2
        elif a == 1:  # quick meal
            if self._pantry >= 1:
                self._pantry -= 1
                self._hunger = self._clip(self._hunger - 2, 0, 10)
                self._nutrition = self._clip(self._nutrition + 1, 0, 10)
            else:
                invalid_penalty = 0.2
        elif a == 2:  # grocery run
            if self._budget >= 6:
                self._budget -= 6
                self._pantry = self._clip(self._pantry + 4, 0, 10)
            else:
                invalid_penalty = 0.2
        elif a == 3:  # skip meal
            self._hunger = self._clip(self._hunger + 1, 0, 10)
            self._nutrition = self._clip(self._nutrition - 1, 0, 10)

        self._day += 1

        health_score = self._nutrition - self._hunger
        reward = 0.2 * health_score
        reward += 0.05 * self._pantry
        reward -= 0.03 * max(0, 12 - self._budget)  # pressure from low budget
        reward -= invalid_penalty

        terminated = self._nutrition >= 8 and self._hunger <= 2 and self._day >= 7
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
        # day, hunger, nutrition, budget_units, pantry
        return MultiDiscreteSpace(nvec=[self._max_days + 1, 11, 11, 51, 11])

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(n=4, start=0)

    def render(self) -> RenderResult:
        text = (
            f"day={self._day}/{self._max_days} "
            f"hunger={self._hunger} "
            f"nutrition={self._nutrition} "
            f"budget={self._budget} "
            f"pantry={self._pantry}  "
            "actions[0=balanced 1=quick 2=grocery 3=skip]"
        )
        return RenderResult(mode="ansi", text=text)

    def sample_action(self) -> int:
        return self._rng.randint(0, 3)

    def _obs(self) -> list[int]:
        return [self._day, self._hunger, self._nutrition, self._budget, self._pantry]

    def _info(self) -> dict[str, Any]:
        return {
            "day": self._day,
            "hunger": self._hunger,
            "nutrition": self._nutrition,
            "budget": self._budget,
            "pantry": self._pantry,
            "max_days": self._max_days,
        }

    @staticmethod
    def _clip(value: int, low: int, high: int) -> int:
        return max(low, min(high, value))


class MealPlanningFactory(rlbridgeEnvironmentFactory):
    """Factory for predefined meal-planning simulators."""

    def __init__(
        self,
        env_id: str = "MealPlanning-Sim-v0",
        *,
        max_days: int = 14,
        start_budget: int = 30,
        tags: Optional[list[str]] = None,
        description: str = "",
    ) -> None:
        self._env_id = env_id
        self._max_days = max_days
        self._start_budget = start_budget
        self._tags = tags or ["meal-planning", "health", "budget", "simulated", "discrete"]
        self._description = description or (
            "Meal planning simulator with pantry usage and grocery restocking. "
            "Balance hunger, nutrition, and budget across multiple days."
        )

    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            env_id=self._env_id,
            description=self._description,
            tags=self._tags,
            namespace="meal_planning",
            render_modes=["ansi"],
            max_episode_steps=self._max_days,
            reward_threshold=2.0,
            suggested_hyperparameters=SuggestedHyperparameters(
                agent_type="ppo",
                n_episodes=1500,
                max_steps=self._max_days,
                gamma=0.99,
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
    ) -> MealPlanningEnvironment:
        _ = render_mode
        return MealPlanningEnvironment(
            max_days=int(kwargs.get("max_days", self._max_days)),
            start_budget=int(kwargs.get("start_budget", self._start_budget)),
        )


MEAL_PLANNING_SIM_V0 = MealPlanningFactory()
ALL_MEAL_PLANNING_FACTORIES: list[MealPlanningFactory] = [MEAL_PLANNING_SIM_V0]
