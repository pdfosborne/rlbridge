"""
Environment wrapper classes for the RLIP MCP plugin.

_ShapedEnv  – injects a sub-goal similarity bonus into step() rewards.
_LangStateEnv – replaces raw observations with their natural-language translations.
"""

from __future__ import annotations

from typing import Any


class _ShapedEnv:
    """
    Thin wrapper around an RLIP environment that injects a sub-goal similarity
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
        bonus: float,
        threshold: float,
        translator: Any,
        env_id: str,
        sub_goal_languages: list[str] | None = None,
    ) -> None:
        self._env = env
        self._bonus = bonus
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

    def _ensure_encoder(self) -> None:
        if self._encoder is not None:
            return
        from ..instruction_following import TextEncoder  # noqa: PLC0415
        enc = TextEncoder()
        enc.fit(self._all_sub_goal_languages)
        self._encoder = enc
        self._sub_goal_vecs = [enc.encode(lg) for lg in self._all_sub_goal_languages]

    def reset(self, seed: Any = None, options: Any = None) -> Any:
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        # Compute max similarity across all sub-goal descriptions and inject bonus.
        try:
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
