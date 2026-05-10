"""
Gymnasium adapter – wraps any gymnasium.Env as an RLIPEnvironment.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces

from ..protocol.messages import (
    EnvironmentInfo,
    RenderResult,
    ResetResult,
    SpaceDescription,
    StepResult,
)
from .base import RLIPEnvironment, RLIPEnvironmentFactory
from .utils import (
    numpy_to_python,
    parse_action,
    rgb_array_to_base64_png,
    space_to_description,
)


class GymnasiumEnvironment(RLIPEnvironment):
    """Thread-safe RLIP wrapper around a gymnasium.Env instance."""

    def __init__(self, env: gym.Env) -> None:
        self._env = env
        self._lock = threading.Lock()
        self._initialized = False
        self._obs_space_desc: SpaceDescription = space_to_description(env.observation_space)
        self._act_space_desc: SpaceDescription = space_to_description(env.action_space)

    # ── Life-cycle ────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        with self._lock:
            obs, info = self._env.reset(seed=seed, options=options or {})
            self._initialized = True
            return ResetResult(
                observation=numpy_to_python(obs),
                info=numpy_to_python(info),
            )

    def step(self, action: Any) -> StepResult:
        with self._lock:
            if not self._initialized:
                from ..protocol.constants import ErrorCodes
                from ..server.exceptions import RLIPError
                raise RLIPError(
                    code=ErrorCodes.ENV_NOT_INITIALIZED,
                    message="Call reset() before step()",
                )
            parsed_action = parse_action(action, self._env.action_space)
            obs, reward, terminated, truncated, info = self._env.step(parsed_action)
            return StepResult(
                observation=numpy_to_python(obs),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=numpy_to_python(info),
            )

    def close(self) -> None:
        with self._lock:
            self._env.close()
            self._initialized = False

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def observation_space(self) -> SpaceDescription:
        return self._obs_space_desc

    @property
    def action_space(self) -> SpaceDescription:
        return self._act_space_desc

    def render(self) -> RenderResult:
        with self._lock:
            render_mode = getattr(self._env, "render_mode", None)
            frame = self._env.render()

            if render_mode == "rgb_array" and frame is not None:
                import numpy as np
                arr = frame if hasattr(frame, "shape") else None
                if arr is not None:
                    h, w = arr.shape[:2]
                    return RenderResult(
                        mode="rgb_array",
                        data=rgb_array_to_base64_png(arr),
                        width=w,
                        height=h,
                    )

            if render_mode == "ansi" and frame is not None:
                return RenderResult(mode="ansi", text=str(frame))

            return RenderResult(mode=render_mode or "unknown", text=str(frame))

    def sample_action(self) -> Any:
        with self._lock:
            return numpy_to_python(self._env.action_space.sample())


class GymnasiumFactory(RLIPEnvironmentFactory):
    """Factory that wraps a registered Gymnasium environment ID."""

    def __init__(
        self,
        env_id: str,
        *,
        description: str = "",
        tags: Optional[list[str]] = None,
    ) -> None:
        self._env_id = env_id
        self._description = description
        self._tags: list[str] = tags or []
        # Eagerly fetch spec metadata
        try:
            spec = gym.spec(env_id)
            self._spec = spec
        except gym.error.Error:
            self._spec = None

    @property
    def env_info(self) -> EnvironmentInfo:
        spec = self._spec
        render_modes: list[str] = []
        max_episode_steps: Optional[int] = None
        reward_threshold: Optional[float] = None
        version = ""

        if spec is not None:
            render_modes = list(getattr(spec, "render_modes", []) or [])
            max_episode_steps = getattr(spec, "max_episode_steps", None)
            reward_threshold = getattr(spec, "reward_threshold", None)
            version = str(getattr(spec, "version", ""))

        return EnvironmentInfo(
            env_id=self._env_id,
            description=self._description or (spec.name if spec else self._env_id),
            version=version,
            tags=self._tags,
            reward_threshold=reward_threshold,
            max_episode_steps=max_episode_steps,
            namespace="gymnasium",
            render_modes=render_modes,
        )

    def create(
        self,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> GymnasiumEnvironment:
        env = gym.make(self._env_id, render_mode=render_mode, **kwargs)
        return GymnasiumEnvironment(env)
