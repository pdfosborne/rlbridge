"""
Environment wrapper classes for the rlbridge MCP plugin.

_ShapedEnv  - injects a sub-goal similarity bonus into step() rewards.
_LangStateEnv - replaces raw observations with their natural-language translations.
"""

from __future__ import annotations

from typing import Any, Optional


class _ShapedEnv:
    """
    Thin wrapper around an rlbridge environment that injects a sub-goal similarity
    bonus into every ``step()`` return value.

    The cosine similarity between the current observation's language description
    and **each** of the *sub_goal_languages* is computed at every step.  When
    the maximum similarity across the set meets or exceeds *threshold* a bonus
    of *bonus* is added to the reward.  Supporting multiple sub-goal languages
    means that any near-identical state to the primary match also triggers the
    shaped reward.

    All other attributes (``reset``, ``close``, ``action_space``, …) are
    forwarded to the wrapped environment unchanged.
    """

    def __init__(
        self,
        env: Any,
        sub_goal_language: str,
        bonus: Optional[float],
        threshold: float,
        translator: Any,
        env_id: str,
        sub_goal_languages: list[str] | None = None,
        encoder_factory: Any = None,
    ) -> None:
        self._env = env
        self._bonus = bonus  # None → auto-scale on first step
        self._threshold = threshold
        self._env_id = env_id
        # Full set of sub-goal language descriptions (primary + extras),
        # de-duplicated while preserving order.
        seen_sg: set[str] = set()
        self._all_sub_goal_languages: list[str] = []
        for lg in [sub_goal_language] + (sub_goal_languages or []):
            if lg not in seen_sg:
                seen_sg.add(lg)
                self._all_sub_goal_languages.append(lg)
        # Resolve language translator
        from ..language_translation import get_translator  # noqa: PLC0415
        from ..language_translation.base import LanguageTranslator  # noqa: PLC0415
        if isinstance(translator, LanguageTranslator):
            self._translator = translator
        else:
            self._translator = get_translator(env_id)
        # Encoder fitted lazily on first step
        self._encoder: Any = None
        self._sub_goal_vecs: list[Any] = []
        self._episode_sub_goal_reached: bool = False
        self._encoder_factory = encoder_factory

    def _ensure_encoder(self) -> None:
        if self._encoder is not None:
            return
        from ..instruction_following import TextEncoder, scale_sub_goal_bonus  # noqa: PLC0415
        enc = self._encoder_factory() if self._encoder_factory is not None else TextEncoder()
        enc.fit(self._all_sub_goal_languages)
        self._encoder = enc
        self._sub_goal_vecs = [enc.encode(lg) for lg in self._all_sub_goal_languages]
        # Auto-scale bonus: max_reward / (100 × n_sub_goals).
        if self._bonus is None:
            self._bonus = scale_sub_goal_bonus(
                self._env, n_instructions=len(self._all_sub_goal_languages)
            )

    def reset(self, seed: Any = None, options: Any = None) -> Any:
        self._episode_sub_goal_reached = False
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        # Compute max similarity across all sub-goal descriptions and inject bonus.
        # First-visit semantics: bonus applied at most once per episode.
        try:
            if not self._episode_sub_goal_reached:
                self._ensure_encoder()
                obs = result.observation if hasattr(result, "observation") else result.get("observation")
                if self._translator and obs is not None:
                    lang = self._translator.translate(obs)
                    obs_vec = self._encoder.encode(lang)
                    sim = max(
                        float(self._encoder.cosine_similarity(obs_vec, sg_vec))
                        for sg_vec in self._sub_goal_vecs
                    )
                    if sim >= self._threshold:
                        self._episode_sub_goal_reached = True
                        if hasattr(result, "reward"):
                            object.__setattr__(result, "reward", result.reward + self._bonus)
                        elif isinstance(result, dict):
                            result = dict(result)
                            result["reward"] = result.get("reward", 0.0) + self._bonus
        except Exception:
            pass  # never crash the training loop over shaping
        return result

    def close(self) -> None:
        self._env.close()

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def env_id(self) -> str:
        return self._env_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class _SequentialShapedEnv:
    """
    Reward-shaping wrapper for ordered multi-step instructions.

    Similarity is computed only against the currently active stage's language
    set. When the active stage is reached, bonus is applied once and the
    wrapper advances to the next stage.
    """

    def __init__(
        self,
        env: Any,
        stage_languages: list[list[str]],
        bonus: Optional[float],
        threshold: float,
        translator: Any,
        env_id: str,
        encoder_factory: Any = None,
    ) -> None:
        self._env = env
        self._bonus = bonus
        self._threshold = threshold
        self._env_id = env_id
        self._stage_languages: list[list[str]] = []
        for stage in stage_languages:
            seen: set[str] = set()
            ordered: list[str] = []
            for lg in stage:
                k = str(lg).strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    ordered.append(str(lg))
            if ordered:
                self._stage_languages.append(ordered)

        from ..language_translation import get_translator  # noqa: PLC0415
        from ..language_translation.base import LanguageTranslator  # noqa: PLC0415
        if isinstance(translator, LanguageTranslator):
            self._translator = translator
        else:
            self._translator = get_translator(env_id)

        self._encoders: list[Any] = []
        self._stage_vecs: list[list[Any]] = []
        self._current_stage: int = 0
        self._encoder_factory = encoder_factory

    def _ensure_encoders(self) -> None:
        if self._encoders:
            return
        from ..instruction_following import TextEncoder, scale_sub_goal_bonus  # noqa: PLC0415

        for stage in self._stage_languages:
            enc = self._encoder_factory() if self._encoder_factory is not None else TextEncoder()
            enc.fit(stage)
            self._encoders.append(enc)
            self._stage_vecs.append([enc.encode(lg) for lg in stage])

        if self._bonus is None:
            self._bonus = scale_sub_goal_bonus(
                self._env, n_instructions=max(1, len(self._stage_languages))
            )

    def reset(self, seed: Any = None, options: Any = None) -> Any:
        self._current_stage = 0
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        try:
            if self._current_stage >= len(self._stage_languages):
                return result
            self._ensure_encoders()
            obs = result.observation if hasattr(result, "observation") else result.get("observation")
            if self._translator and obs is not None:
                lang = self._translator.translate(obs)
                enc = self._encoders[self._current_stage]
                obs_vec = enc.encode(lang)
                sim = max(
                    float(enc.cosine_similarity(obs_vec, sg_vec))
                    for sg_vec in self._stage_vecs[self._current_stage]
                )
                if sim >= self._threshold:
                    if hasattr(result, "reward"):
                        object.__setattr__(result, "reward", result.reward + self._bonus)
                    elif isinstance(result, dict):
                        result = dict(result)
                        result["reward"] = result.get("reward", 0.0) + self._bonus
                    # Inject sub_goal info into info dict so renderers can highlight it
                    try:
                        if hasattr(result, "info"):
                            new_info = dict(result.info or {})
                            new_info["sub_goal_reached"] = True
                            new_info["sub_goal_similarity"] = round(float(sim), 4)
                            new_info["instruction_index"] = int(self._current_stage)
                            object.__setattr__(result, "info", new_info)
                        elif isinstance(result, dict):
                            result["sub_goal_reached"] = True
                            result["sub_goal_similarity"] = round(float(sim), 4)
                            result["instruction_index"] = int(self._current_stage)
                    except Exception:
                        pass
                    self._current_stage += 1
        except Exception:
            pass
        return result

    def close(self) -> None:
        self._env.close()

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def env_id(self) -> str:
        return self._env_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class _LangStateEnv:
    """
    Environment wrapper that replaces raw observations with their
    natural-language translations at every ``reset()`` and ``step()``.

    When the translator returns an empty string or raises an exception the
    raw observation is returned unchanged as a safe fallback, so training
    always continues.

    Stack order when used together with ``_ShapedEnv``::

        _LangStateEnv(
            _ShapedEnv(base_env, ...)   ← injects reward bonus using raw obs
        )                                ← agent then sees language string obs

    The ``action_space``, ``env_id``, ``close``, and all other attributes
    are forwarded transparently to the wrapped environment.
    """

    def __init__(self, env: Any, translator: Any, env_id: str) -> None:
        self._env = env
        self._env_id = env_id
        from ..language_translation import get_translator  # noqa: PLC0415
        from ..language_translation.base import LanguageTranslator  # noqa: PLC0415
        if isinstance(translator, LanguageTranslator):
            self._translator: Any = translator
        else:
            self._translator = get_translator(env_id)

    def _translate(self, obs: Any) -> Any:
        """Return the language description of *obs*, or *obs* on failure."""
        if self._translator is None:
            return obs
        try:
            lang = self._translator.translate(obs)
            return lang if lang else obs
        except Exception:
            return obs

    def _apply_to_result(self, result: Any, key: str, translated: Any) -> Any:
        """Replace *key* in a Pydantic model or dict result."""
        if hasattr(result, key):
            object.__setattr__(result, key, translated)
        elif isinstance(result, dict):
            result = dict(result)
            result[key] = translated
        return result

    def reset(self, seed: Any = None, options: Any = None) -> Any:
        result = self._env.reset(seed=seed, options=options)
        if hasattr(result, "observation"):
            translated = self._translate(result.observation)
            result = self._apply_to_result(result, "observation", translated)
        elif isinstance(result, dict) and "observation" in result:
            translated = self._translate(result["observation"])
            result = dict(result)
            result["observation"] = translated
        else:
            # The result itself is the raw observation
            result = self._translate(result)
        return result

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        if hasattr(result, "observation"):
            translated = self._translate(result.observation)
            result = self._apply_to_result(result, "observation", translated)
        elif isinstance(result, dict) and "observation" in result:
            translated = self._translate(result["observation"])
            result = dict(result)
            result["observation"] = translated
        return result

    def close(self) -> None:
        self._env.close()

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def env_id(self) -> str:
        return self._env_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)
