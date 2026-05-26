"""
rlbridge Simulated Smart Home Control Environment
================================================
Discrete smart-home control task balancing comfort and energy cost.

State
-----
Observation is a list:
    [hour, indoor_temp_bin, occupancy, energy_bin]

Actions
-------
    0 - HVAC off
    1 - HVAC cool
    2 - HVAC heat
    3 - Eco mode (mild correction, lower energy draw)

Goal
----
Keep indoor temperature comfortable while minimising energy use during a
24-hour cycle.
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


class SmartHomeEnvironment(rlbridgeEnvironment):
    """Simple thermostat-like control simulator."""

    def __init__(
        self,
        *,
        max_hours: int = 24,
        comfort_target: float = 22.0,
    ) -> None:
        self._max_hours = max_hours
        self._comfort_target = comfort_target

        self._rng = random.Random()
        self._initialized = False

        self._hour = 0
        self._indoor_temp = 22.0
        self._energy = 0.0
        self._occupancy = 1

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        if seed is not None:
            self._rng.seed(seed)

        opts = options or {}
        self._hour = int(opts.get("start_hour", 0))
        self._indoor_temp = float(opts.get("start_temp", 22.0))
        self._energy = float(opts.get("start_energy", 0.0))
        self._occupancy = int(opts.get("start_occupancy", 1))

        self._hour = self._clip_int(self._hour, 0, self._max_hours)
        self._indoor_temp = self._clip_float(self._indoor_temp, 10.0, 35.0)
        self._energy = self._clip_float(self._energy, 0.0, 100.0)
        self._occupancy = self._clip_int(self._occupancy, 0, 1)

        self._initialized = True
        return ResetResult(observation=self._obs(), info=self._info())

    def step(self, action: Any) -> StepResult:
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")

        a = int(action)
        if a not in (0, 1, 2, 3):
            raise ValueError(f"Action must be 0..3, got {a}")

        outdoor_temp = self._outdoor_temp(self._hour)
        self._occupancy = self._occupancy_for_hour(self._hour)

        # Passive thermal drift toward outdoor temperature.
        drift = 0.2 * (outdoor_temp - self._indoor_temp)
        self._indoor_temp += drift

        # Control action effect and energy usage.
        energy_use = 0.0
        if a == 1:  # cool
            self._indoor_temp -= 1.2
            energy_use = 2.0
        elif a == 2:  # heat
            self._indoor_temp += 1.2
            energy_use = 2.0
        elif a == 3:  # eco mode
            if self._indoor_temp > self._comfort_target:
                self._indoor_temp -= 0.5
            elif self._indoor_temp < self._comfort_target:
                self._indoor_temp += 0.5
            energy_use = 1.0

        self._indoor_temp = self._clip_float(self._indoor_temp, 10.0, 35.0)
        self._energy += energy_use
        self._energy = self._clip_float(self._energy, 0.0, 100.0)
        self._hour += 1

        comfort_error = abs(self._indoor_temp - self._comfort_target)
        comfort_weight = 1.0 if self._occupancy == 1 else 0.4
        reward = -(comfort_weight * comfort_error + 0.25 * energy_use)

        terminated = False
        truncated = self._hour >= self._max_hours

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
        # hour, indoor_temp_bin, occupancy, energy_bin
        return MultiDiscreteSpace(nvec=[self._max_hours + 1, 26, 2, 21])

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(n=4, start=0)

    def render(self) -> RenderResult:
        text = (
            f"hour={self._hour}/{self._max_hours} "
            f"temp={self._indoor_temp:.1f}C "
            f"outdoor={self._outdoor_temp(min(self._hour, self._max_hours - 1)):.1f}C "
            f"occupancy={self._occupancy} "
            f"energy={self._energy:.1f}  "
            "actions[0=off 1=cool 2=heat 3=eco]"
        )
        return RenderResult(mode="ansi", text=text)

    def sample_action(self) -> int:
        return self._rng.randint(0, 3)

    def _obs(self) -> list[int]:
        temp_bin = self._clip_int(int(round(self._indoor_temp)) - 10, 0, 25)
        energy_bin = self._clip_int(int(self._energy / 5.0), 0, 20)
        return [self._hour, temp_bin, self._occupancy, energy_bin]

    def _info(self) -> dict[str, Any]:
        return {
            "hour": self._hour,
            "indoor_temp": round(self._indoor_temp, 3),
            "outdoor_temp": round(self._outdoor_temp(min(self._hour, self._max_hours - 1)), 3),
            "occupancy": self._occupancy,
            "energy": round(self._energy, 3),
            "max_hours": self._max_hours,
            "comfort_target": self._comfort_target,
        }

    def _outdoor_temp(self, hour: int) -> float:
        # Piecewise profile: cool night, warm afternoon.
        h = hour % 24
        if 0 <= h <= 5:
            return 16.0
        if 6 <= h <= 10:
            return 20.0
        if 11 <= h <= 16:
            return 28.0
        if 17 <= h <= 21:
            return 23.0
        return 18.0

    @staticmethod
    def _occupancy_for_hour(hour: int) -> int:
        h = hour % 24
        return 0 if 9 <= h <= 16 else 1

    @staticmethod
    def _clip_int(value: int, low: int, high: int) -> int:
        return max(low, min(high, value))

    @staticmethod
    def _clip_float(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))


class SmartHomeFactory(rlbridgeEnvironmentFactory):
    """Factory for predefined smart-home control simulators."""

    def __init__(
        self,
        env_id: str = "SmartHome-Control-v0",
        *,
        max_hours: int = 24,
        comfort_target: float = 22.0,
        tags: Optional[list[str]] = None,
        description: str = "",
    ) -> None:
        self._env_id = env_id
        self._max_hours = max_hours
        self._comfort_target = comfort_target
        self._tags = tags or ["smart-home", "control", "energy", "simulated", "discrete"]
        self._description = description or (
            "Smart-home HVAC control simulator balancing occupant comfort and "
            "energy consumption across a daily cycle."
        )

    @property
    def env_info(self) -> EnvironmentInfo:
        return EnvironmentInfo(
            env_id=self._env_id,
            description=self._description,
            tags=self._tags,
            namespace="smart_home",
            render_modes=["ansi"],
            max_episode_steps=self._max_hours,
            reward_threshold=-10.0,
            suggested_hyperparameters=SuggestedHyperparameters(
                agent_type="tabular_q",
                n_episodes=3000,
                max_steps=self._max_hours,
                alpha=0.1,
                gamma=0.98,
                epsilon=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.998,
                sub_goal_threshold=0.5,
                top_k=3,
                min_episode_visits=2,
            ),
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> SmartHomeEnvironment:
        _ = render_mode
        return SmartHomeEnvironment(
            max_hours=int(kwargs.get("max_hours", self._max_hours)),
            comfort_target=float(kwargs.get("comfort_target", self._comfort_target)),
        )


SMART_HOME_CONTROL_V0 = SmartHomeFactory()
ALL_SMART_HOME_FACTORIES: list[SmartHomeFactory] = [SMART_HOME_CONTROL_V0]
