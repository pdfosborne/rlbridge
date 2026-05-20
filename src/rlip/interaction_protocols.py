"""
RLIP Interaction Protocols
============================
Defines named, reusable forms of agent–environment interaction.

Each protocol is a callable object (or plain function) that accepts an
:class:`~rlip.environments.base.RLIPEnvironment` instance and drives it
according to a specific interaction pattern.

Available protocols
-------------------
- :class:`SingleStepProtocol`  – manual one-step-at-a-time control
- :class:`RandomEpisodeProtocol` – full episode with a uniform-random policy
- :class:`GreedyEpisodeProtocol` – full episode using a caller-supplied policy fn
- :class:`MultiEpisodeProtocol` – wrapper that repeats any protocol N times
- :func:`run_protocol`         – convenience helper to run a named protocol by string

All protocols return an :class:`InteractionResult` dataclass.

Language translation
--------------------
All protocols accept a ``translate`` parameter that maps raw observations to
natural-language descriptions via :mod:`rlip.language_translation`.

    translate=True          – auto-lookup translator for the environment
    translate=False / None  – no translation (default)
    translate=<translator>  – use a specific :class:`~rlip.language_translation.LanguageTranslator`

When enabled, each :class:`StepRecord` gains a ``language_obs`` string field
alongside the raw ``observation``.

Example::

    result = RandomEpisodeProtocol(max_steps=50, seed=0, translate=True)(env)
    for rec in result.episodes[0].history:
        print(rec.language_obs)

Example
-------
    from rlip.interaction_protocols import RandomEpisodeProtocol
    from rlip.environments.registry import registry

    factory = registry.get("CartPole-v1")
    env = factory.create()
    result = RandomEpisodeProtocol(max_steps=500, seed=0)(env)
    print(result)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Union, runtime_checkable

from .language_translation.base import LanguageTranslator

# Convenience type for the translate= parameter accepted by all protocols.
_TranslateArg = Union[bool, LanguageTranslator, None]


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """One (action, obs, reward, terminated, truncated) transition."""
    step: int
    action: Any
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
    language_obs: Optional[str] = field(default=None, repr=False)
    """Natural-language description of *observation*, or ``None`` if translation
    was not requested or no translator is registered for this environment."""


@dataclass
class EpisodeResult:
    """Summary of a single episode run."""
    episode: int
    steps: int
    total_reward: float
    end_reason: str                        # "terminated" | "truncated" | "max_steps"
    history: list[StepRecord] = field(default_factory=list, repr=False)
    seed: Optional[int] = field(default=None, repr=False)

    @property
    def done(self) -> bool:
        return self.end_reason in ("terminated", "truncated")


@dataclass
class InteractionResult:
    """Aggregate result across one or more episodes."""
    protocol_name: str
    env_id: str
    episodes: list[EpisodeResult] = field(default_factory=list)

    # ── Convenience aggregates ────────────────────────────────────────────────

    @property
    def total_steps(self) -> int:
        return sum(e.steps for e in self.episodes)

    @property
    def total_reward(self) -> float:
        return sum(e.total_reward for e in self.episodes)

    @property
    def mean_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return self.total_reward / len(self.episodes)

    @property
    def mean_steps(self) -> float:
        if not self.episodes:
            return 0.0
        return self.total_steps / len(self.episodes)

    def __str__(self) -> str:
        ep = len(self.episodes)
        return (
            f"InteractionResult[{self.protocol_name}] env={self.env_id} "
            f"episodes={ep} mean_reward={self.mean_reward:.3f} "
            f"mean_steps={self.mean_steps:.1f}"
        )


# ── Environment interface expected by all protocols ───────────────────────────

@runtime_checkable
class _EnvLike(Protocol):
    """Structural type expected by interaction protocols."""

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Any: ...
    def step(self, action: Any) -> Any: ...
    def close(self) -> None: ...

    @property
    def action_space(self) -> Any: ...


# ── Action samplers ───────────────────────────────────────────────────────────

def _make_sampler(env: _EnvLike, seed: Optional[int]) -> Callable[[], Any]:
    """
    Build a reproducible random action sampler for *env*.

    Tries the spaces described in ``env.action_space`` first; falls back to
    calling ``env.sample_action()`` if available; otherwise returns 0.
    """
    rng = random.Random(seed)

    space = env.action_space
    space_type = getattr(space, "type", None)

    # Pydantic SpaceDescription objects (RLIP's own format)
    if space_type == "Discrete":
        n: int = space.n
        start: int = getattr(space, "start", 0)
        return lambda: rng.randint(start, start + n - 1)

    if space_type == "MultiBinary":
        n_bits: int = space.n
        return lambda: [rng.randint(0, 1) for _ in range(n_bits)]

    # Dict-style space description (raw dicts from dispatcher results)
    if isinstance(space, dict):
        dtype = space.get("type", "")
        if dtype == "Discrete":
            n = space["n"]
            start = space.get("start", 0)
            return lambda: rng.randint(start, start + n - 1)

    # Try the environment's own sampler
    if hasattr(env, "sample_action"):
        def _safe_env_sample() -> Any:
            try:
                return env.sample_action()
            except NotImplementedError:
                return 0
            except Exception:
                # Some environments only support sampling after reset().
                return 0
        return _safe_env_sample

    # Last resort
    return lambda: 0


# ── Protocol base ─────────────────────────────────────────────────────────────

class _BaseProtocol:
    name: str = "base"

    def _env_id(self, env: _EnvLike) -> str:
        return getattr(env, "env_id", type(env).__name__)

    def _resolve_translator(
        self,
        translate: _TranslateArg,
        env_id: str,
    ) -> Optional[LanguageTranslator]:
        """Return the translator to use, or *None* if translation is disabled."""
        if translate is False or translate is None:
            return None
        if isinstance(translate, LanguageTranslator):
            return translate
        # translate=True — auto-lookup
        from .language_translation import get_translator
        return get_translator(env_id)

    def _get_effective_translator(
        self,
        translate: _TranslateArg,
        env_id: str,
        llm_fn: Optional[Callable[[str], str]] = None,
        llm_refine_threshold: int = 20,
        llm_env_context: str = "",
    ) -> Optional[LanguageTranslator]:
        """
        Like :meth:`_resolve_translator` but wraps the result with a
        :class:`~rlip.language_translation.generator.GeneratedTranslator` when
        *llm_fn* is provided.

        The ``GeneratedTranslator`` layer adds LLM-based fallback for states
        the registered translator cannot describe (returns ``""``), caches
        those answers, and automatically re-runs rule synthesis when the cache
        reaches *llm_refine_threshold* entries.  The wrapper is cached per
        env_id on this protocol instance so descriptions accumulate across
        multiple ``__call__`` invocations.
        """
        base = self._resolve_translator(translate, env_id)
        if llm_fn is None:
            return base
        from .language_translation.generator import GeneratedTranslator
        if not hasattr(self, "_llm_translators"):
            self._llm_translators: dict[str, LanguageTranslator] = {}
        if env_id not in self._llm_translators:
            self._llm_translators[env_id] = GeneratedTranslator(
                llm_fn=llm_fn,
                env_id=env_id,
                env_context=llm_env_context,
                refine_threshold=llm_refine_threshold,
                base_translator=base,
            )
        return self._llm_translators[env_id]

    def __call__(self, env: _EnvLike) -> InteractionResult:
        raise NotImplementedError

    def __repr__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        kv = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
        return f"{type(self).__name__}({kv})"


# ── 1. SingleStepProtocol ─────────────────────────────────────────────────────

class SingleStepProtocol(_BaseProtocol):
    """
    Execute exactly one action in the environment.

    Useful for manual, interactive, or tool-driven control where the caller
    decides the action at each turn.

    Parameters
    ----------
    action:
        The action to execute.  If *None* a random action is sampled.
    seed:
        Seed used both for the environment reset and random action sampling.
    reset_first:
        If *True* (default), call ``env.reset()`` before stepping.
    record_history:
        If *True*, include the full :class:`StepRecord` in the result.
    """

    name = "single_step"

    def __init__(
        self,
        action: Any = None,
        seed: Optional[int] = None,
        reset_first: bool = True,
        record_history: bool = True,
        translate: _TranslateArg = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        llm_refine_threshold: int = 20,
        llm_env_context: str = "",
    ) -> None:
        self.action = action
        self.seed = seed
        self.reset_first = reset_first
        self.record_history = record_history
        self.translate = translate
        self.llm_fn = llm_fn
        self.llm_refine_threshold = llm_refine_threshold
        self.llm_env_context = llm_env_context

    def __call__(self, env: _EnvLike) -> InteractionResult:
        env_id = self._env_id(env)
        result = InteractionResult(protocol_name=self.name, env_id=env_id)
        translator = self._get_effective_translator(
            self.translate, env_id, self.llm_fn, self.llm_refine_threshold, self.llm_env_context
        )

        if self.reset_first:
            env.reset(seed=self.seed)

        act = self.action if self.action is not None else _make_sampler(env, self.seed)()
        step_out = env.step(act)

        # Support both StepResult (dataclass/Pydantic) and plain dicts
        obs        = _get(step_out, "observation")
        reward     = float(_get(step_out, "reward", 0.0))
        terminated = bool(_get(step_out, "terminated", False))
        truncated  = bool(_get(step_out, "truncated", False))
        info       = _get(step_out, "info", {}) or {}

        language_obs = translator.translate(obs, action_history=[act]) if translator else None

        end_reason = (
            "terminated" if terminated
            else "truncated" if truncated
            else "running"
        )
        rec = StepRecord(
            step=1, action=act, observation=obs,
            reward=reward, terminated=terminated,
            truncated=truncated, info=info,
            language_obs=language_obs,
        )
        ep = EpisodeResult(
            episode=1, steps=1, total_reward=reward, end_reason=end_reason,
            history=[rec] if self.record_history else [],
        )
        result.episodes.append(ep)
        return result


# ── 2. RandomEpisodeProtocol ──────────────────────────────────────────────────

class RandomEpisodeProtocol(_BaseProtocol):
    """
    Run a full episode using a uniform-random policy.

    Parameters
    ----------
    max_steps:
        Hard cap on episode length.
    seed:
        Seed for environment reset and action sampling.
    record_history:
        Store the full step-by-step trajectory in the result.
    """

    name = "random_episode"

    def __init__(
        self,
        max_steps: int = 200,
        seed: Optional[int] = None,
        record_history: bool = False,
        translate: _TranslateArg = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        llm_refine_threshold: int = 20,
        llm_env_context: str = "",
    ) -> None:
        self.max_steps = max_steps
        self.seed = seed
        self.record_history = record_history
        self.translate = translate
        self.llm_fn = llm_fn
        self.llm_refine_threshold = llm_refine_threshold
        self.llm_env_context = llm_env_context

    def __call__(self, env: _EnvLike) -> InteractionResult:
        env_id = self._env_id(env)
        result = InteractionResult(protocol_name=self.name, env_id=env_id)
        translator = self._get_effective_translator(
            self.translate, env_id, self.llm_fn, self.llm_refine_threshold, self.llm_env_context
        )
        env.reset(seed=self.seed)
        sampler = _make_sampler(env, self.seed)
        ep = _run_episode(env, sampler, self.max_steps, episode_n=1,
                          record=self.record_history, translator=translator)
        result.episodes.append(ep)
        return result


# ── 3. GreedyEpisodeProtocol ──────────────────────────────────────────────────

class GreedyEpisodeProtocol(_BaseProtocol):
    """
    Run a full episode using a caller-supplied policy function.

    The policy receives the current observation and returns an action.

    Parameters
    ----------
    policy_fn:
        ``Callable[[obs], action]`` — any function mapping an observation to
        an action.  The observation format matches what ``env.step()`` returns.
    max_steps:
        Hard cap on episode length.
    seed:
        Seed for environment reset.
    record_history:
        Store the full step-by-step trajectory in the result.

    Example
    -------
        def always_push_right(obs):
            return 1  # CartPole: always push right

        result = GreedyEpisodeProtocol(always_push_right, max_steps=500)(env)
    """

    name = "greedy_episode"

    def __init__(
        self,
        policy_fn: Callable[[Any], Any],
        max_steps: int = 200,
        seed: Optional[int] = None,
        record_history: bool = False,
        translate: _TranslateArg = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        llm_refine_threshold: int = 20,
        llm_env_context: str = "",
    ) -> None:
        self.policy_fn = policy_fn
        self.max_steps = max_steps
        self.seed = seed
        self.record_history = record_history
        self.translate = translate
        self.llm_fn = llm_fn
        self.llm_refine_threshold = llm_refine_threshold
        self.llm_env_context = llm_env_context

    def __call__(self, env: _EnvLike) -> InteractionResult:
        env_id = self._env_id(env)
        result = InteractionResult(protocol_name=self.name, env_id=env_id)
        translator = self._get_effective_translator(
            self.translate, env_id, self.llm_fn, self.llm_refine_threshold, self.llm_env_context
        )
        reset_out = env.reset(seed=self.seed)
        obs = _get(reset_out, "observation", reset_out)

        total_reward = 0.0
        history: list[StepRecord] = []
        action_history: list[Any] = []

        for step_n in range(1, self.max_steps + 1):
            action = self.policy_fn(obs)
            step_out = env.step(action)
            action_history.append(action)

            obs        = _get(step_out, "observation", obs)
            reward     = float(_get(step_out, "reward", 0.0))
            terminated = bool(_get(step_out, "terminated", False))
            truncated  = bool(_get(step_out, "truncated", False))
            info       = _get(step_out, "info", {}) or {}

            total_reward += reward
            if self.record_history:
                language_obs = (
                    translator.translate(obs, action_history=action_history)
                    if translator else None
                )
                history.append(StepRecord(
                    step=step_n, action=action, observation=obs,
                    reward=reward, terminated=terminated,
                    truncated=truncated, info=info,
                    language_obs=language_obs,
                ))

            if terminated or truncated:
                end_reason = "terminated" if terminated else "truncated"
                break
        else:
            end_reason = "max_steps"

        result.episodes.append(EpisodeResult(
            episode=1, steps=step_n,
            total_reward=total_reward,
            end_reason=end_reason,
            history=history,
        ))
        return result


# ── 4. MultiEpisodeProtocol ───────────────────────────────────────────────────

class MultiEpisodeProtocol(_BaseProtocol):
    """
    Repeat a base protocol for *n_episodes* and aggregate results.

    Parameters
    ----------
    base:
        Any protocol instance to repeat.
    n_episodes:
        Number of episodes to run.
    base_seed:
        If given, seeds are derived as ``base_seed + episode_index`` for
        reproducibility.  If *None*, each episode uses a fresh random seed.

    Example
    -------
        base = RandomEpisodeProtocol(max_steps=500)
        multi = MultiEpisodeProtocol(base, n_episodes=10, base_seed=0)
        result = multi(env)
        print(result.mean_reward)
    """

    name = "multi_episode"

    def __init__(
        self,
        base: _BaseProtocol,
        n_episodes: int = 10,
        base_seed: Optional[int] = None,
    ) -> None:
        self.base = base
        self.n_episodes = n_episodes
        self.base_seed = base_seed

    def __call__(self, env: _EnvLike) -> InteractionResult:
        result = InteractionResult(
            protocol_name=f"multi_episode[{self.base.name}]",
            env_id=self._env_id(env),
        )

        for i in range(self.n_episodes):
            # Patch seed on the base protocol for each episode
            if self.base_seed is not None and hasattr(self.base, "seed"):
                self.base.seed = self.base_seed + i  # type: ignore[union-attr]

            episode_seed = (self.base_seed + i) if self.base_seed is not None else None
            ep_result = self.base(env)
            if ep_result.episodes:
                ep = ep_result.episodes[0]
                ep.episode = i + 1
                ep.seed = episode_seed
                result.episodes.append(ep)

        return result


# ── Registry of built-in protocols ───────────────────────────────────────────

# ── 5. InstructionFollowingProtocol ──────────────────────────────────────────

class InstructionFollowingProtocol(_BaseProtocol):
    """
    Episode protocol with language-instruction sub-goal shaping.

    After an instruction is matched to an observed state via
    :func:`~rlip.instruction_following.match_instruction`, this protocol
    drives an RL training episode and adds a one-time (or repeatable) bonus
    reward whenever the agent visits a state whose language description has
    a cosine similarity to **any** of the sub-goal language descriptions that
    meets or exceeds *sub_goal_threshold*.

    Parameters
    ----------
    instruction:
        The original natural-language instruction (stored for reference).
    sub_goal_language:
        The primary (highest-scoring) translated language description of the
        matched sub-goal state, used as the primary similarity target.
    sub_goal_observation:
        The raw environment observation that was matched as the sub-goal
        (stored for reference, not used during scoring).
    sub_goal_languages:
        Additional language descriptions that are treated as equivalent
        sub-goals alongside *sub_goal_language*.  All are encoded and the
        maximum cosine similarity across the set is used at each step.
        Populated automatically from :attr:`InstructionMatch.matched_states`
        when using :func:`~rlip.instruction_following.build_sequential_instruction_following_protocol`.
    policy_fn:
        Policy ``Callable[[obs], action]``.  Defaults to random sampling.
    sub_goal_bonus:
        Bonus reward added when the sub-goal similarity threshold is met.
    sub_goal_threshold:
        Cosine similarity threshold (0–1) required to count a step as
        reaching any sub-goal.
    sub_goal_repeatable:
        If *False* (default) the protocol uses **first-visit** semantics:
        the bonus is given exactly once (the first step the threshold is met)
        and similarity is **not** computed for the rest of the episode, so the
        agent receives no sub-goal signal after the first visit and is not
        tempted to linger.
        If *True* the bonus fires on every step the threshold is met.
    max_steps:
        Hard cap on episode length.
    seed:
        Seed for environment reset and random action sampling.
    record_history:
        Store full step trajectory in the result.
    translate:
        Translation source (see module-level docs).  Defaults to *True*
        (auto-lookup).
    """

    name = "instruction_following"

    def __init__(
        self,
        instruction: str,
        sub_goal_language: str,
        sub_goal_observation: Any = None,
        policy_fn: Optional[Callable[[Any], Any]] = None,
        sub_goal_bonus: float = 0.1,
        sub_goal_threshold: float = 0.5,
        sub_goal_repeatable: bool = False,
        max_steps: int = 200,
        seed: Optional[int] = None,
        record_history: bool = True,
        translate: _TranslateArg = True,
        encoder_factory: Optional[Any] = None,
        sub_goal_languages: Optional[list[str]] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        llm_refine_threshold: int = 20,
        llm_env_context: str = "",
        _stats_tracker: Any = None,
        _state_instruction_map: Optional[dict] = None,
    ) -> None:
        self.instruction = instruction
        self.sub_goal_language = sub_goal_language
        self.sub_goal_observation = sub_goal_observation
        self.policy_fn = policy_fn
        self.sub_goal_bonus = sub_goal_bonus
        self.sub_goal_threshold = sub_goal_threshold
        self.sub_goal_repeatable = sub_goal_repeatable
        self.max_steps = max_steps
        self.seed = seed
        self.record_history = record_history
        self.translate = translate
        # encoder_factory is a zero-arg callable returning a BaseEncoder.
        # Defaults to TFIDFEncoder (lazy import to avoid circular deps).
        self._encoder_factory = encoder_factory
        self.llm_fn = llm_fn
        self.llm_refine_threshold = llm_refine_threshold
        self.llm_env_context = llm_env_context
        # Full set of sub-goal language descriptions (primary + extras).
        # De-duplicated while preserving order (primary first).
        _all = [sub_goal_language] + (sub_goal_languages or [])
        seen_langs: set[str] = set()
        self._all_sub_goal_languages: list[str] = []
        for lg in _all:
            if lg not in seen_langs:
                seen_langs.add(lg)
                self._all_sub_goal_languages.append(lg)
        self._encoder: Any = None
        self._sub_goal_vecs: list[Any] = []  # one vector per sub-goal language
        # Optional InstructionCacheEntry (duck-typed) for success-rate tracking.
        self._stats_tracker = _stats_tracker
        # Optional live dict mapping language_description → [instruction, ...]
        # used to annotate steps with other instructions targeting the same state.
        self._state_instruction_map: Optional[dict] = _state_instruction_map

    def __call__(self, env: _EnvLike) -> InteractionResult:
        # Lazy import avoids circular dependency at module load time.
        from .instruction_matching import TextEncoder  # noqa: PLC0415

        env_id = self._env_id(env)
        result = InteractionResult(protocol_name=self.name, env_id=env_id)
        translator = self._get_effective_translator(
            self.translate, env_id, self.llm_fn, self.llm_refine_threshold, self.llm_env_context
        )

        # Build / reuse encoder fitted on instruction + all sub-goal languages.
        if self._encoder is None:
            factory = self._encoder_factory if self._encoder_factory is not None else TextEncoder
            corpus = [self.instruction] + self._all_sub_goal_languages
            self._encoder = factory().fit(corpus)
            self._sub_goal_vecs = [
                self._encoder.encode(lg) for lg in self._all_sub_goal_languages
            ]

        reset_out = env.reset(seed=self.seed)
        obs = _get(reset_out, "observation", reset_out)

        sampler = _make_sampler(env, self.seed)
        policy = self.policy_fn if self.policy_fn is not None else (lambda _obs: sampler())

        total_reward = 0.0
        history: list[StepRecord] = []
        action_history: list[Any] = []
        sub_goal_reached = False

        step_n = 0
        terminated = False
        truncated = False

        for step_n in range(1, self.max_steps + 1):
            action = policy(obs)
            step_out = env.step(action)
            action_history.append(action)

            obs        = _get(step_out, "observation", obs)
            reward     = float(_get(step_out, "reward", 0.0))
            terminated = bool(_get(step_out, "terminated", False))
            truncated  = bool(_get(step_out, "truncated", False))
            info       = dict(_get(step_out, "info", {}) or {})

            # ── Language-based sub-goal shaping ───────────────────────────────
            language_obs: Optional[str] = None
            if translator:
                language_obs = translator.translate(obs, action_history=action_history)

                # Annotate steps that land on states matched to *any* instruction
                # (including instructions other than the currently active one).
                if self._state_instruction_map and language_obs in self._state_instruction_map:
                    associated = self._state_instruction_map[language_obs]
                    if associated:
                        info["associated_instructions"] = list(associated)

                # First-visit semantics: once the sub-goal has been reached,
                # skip similarity computation entirely so the agent receives no
                # signal that would encourage it to linger near the sub-goal.
                # When sub_goal_repeatable=True the check runs every step.
                if not sub_goal_reached or self.sub_goal_repeatable:
                    obs_vec = self._encoder.encode(language_obs)
                    # Max similarity across all sub-goal descriptions.
                    sim = max(
                        self._encoder.cosine_similarity(obs_vec, sg_vec)
                        for sg_vec in self._sub_goal_vecs
                    )
                    info["sub_goal_similarity"] = round(sim, 4)

                    if sim >= self.sub_goal_threshold:
                        reward += self.sub_goal_bonus
                        sub_goal_reached = True
                        info["sub_goal_reached"] = True

            total_reward += reward

            if self.record_history:
                history.append(StepRecord(
                    step=step_n, action=action, observation=obs,
                    reward=reward, terminated=terminated,
                    truncated=truncated, info=info,
                    language_obs=language_obs,
                ))

            if terminated or truncated:
                break

        end_reason = (
            "terminated" if terminated
            else "truncated" if truncated
            else "max_steps"
        )
        result.episodes.append(EpisodeResult(
            episode=1,
            steps=step_n,
            total_reward=total_reward,
            end_reason=end_reason,
            history=history,
        ))

        # Record episode outcome into the instruction cache entry (if provided).
        if self._stats_tracker is not None:
            self._stats_tracker.record_episode(sub_goal_reached)

        return result


_PROTOCOL_REGISTRY: dict[str, type[_BaseProtocol]] = {
    "single_step":         SingleStepProtocol,
    "random_episode":      RandomEpisodeProtocol,
    "greedy_episode":      GreedyEpisodeProtocol,
    "multi_episode":       MultiEpisodeProtocol,
    "instruction_following": InstructionFollowingProtocol,
}


def list_protocols() -> list[str]:
    """Return the names of all registered interaction protocols."""
    return sorted(_PROTOCOL_REGISTRY)


def run_protocol(
    name: str,
    env: _EnvLike,
    **kwargs: Any,
) -> InteractionResult:
    """
    Instantiate and run a named interaction protocol.

    Parameters
    ----------
    name:
        One of the protocol names returned by :func:`list_protocols`.
    env:
        An RLIP environment instance.
    **kwargs:
        Forwarded to the protocol's ``__init__``.

    Raises
    ------
    KeyError
        If *name* is not a registered protocol.

    Example
    -------
        result = run_protocol("random_episode", env, max_steps=500, seed=42)
    """
    cls = _PROTOCOL_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown protocol '{name}'. Available: {list_protocols()}"
        )
    return cls(**kwargs)(env)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute- or dict-access that works for both Pydantic models and dicts."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _run_episode(
    env: _EnvLike,
    sampler: Callable[[], Any],
    max_steps: int,
    episode_n: int,
    record: bool,
    translator: Optional[LanguageTranslator] = None,
) -> EpisodeResult:
    total_reward = 0.0
    history: list[StepRecord] = []
    action_history: list[Any] = []

    step_n = 0
    terminated = False
    truncated = False

    for step_n in range(1, max_steps + 1):
        action = sampler()
        step_out = env.step(action)
        action_history.append(action)

        obs        = _get(step_out, "observation")
        reward     = float(_get(step_out, "reward", 0.0))
        terminated = bool(_get(step_out, "terminated", False))
        truncated  = bool(_get(step_out, "truncated", False))
        info       = _get(step_out, "info", {}) or {}

        total_reward += reward
        if record:
            language_obs = (
                translator.translate(obs, action_history=action_history)
                if translator else None
            )
            history.append(StepRecord(
                step=step_n, action=action, observation=obs,
                reward=reward, terminated=terminated,
                truncated=truncated, info=info,
                language_obs=language_obs,
            ))
        if terminated or truncated:
            break

    end_reason = (
        "terminated" if terminated
        else "truncated" if truncated
        else "max_steps"
    )
    return EpisodeResult(
        episode=episode_n,
        steps=step_n,
        total_reward=total_reward,
        end_reason=end_reason,
        history=history,
    )
