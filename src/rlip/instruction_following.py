"""
Instruction Following for RLIP
================================
Connects a natural-language instruction to RL environment training via three
stages:

1. **Exploration** – an interaction protocol (optionally driven by a
   Claude/OpenAI agent via :class:`~rlip.adapters.openai_agent.RLIPAgent`)
   runs the environment and collects a trajectory of raw observations.

2. **Language translation** – each raw observation is passed through a
   :class:`~rlip.language_translation.LanguageTranslator` to produce a
   natural-language description.

3. **Instruction matching** – TF-IDF shortlists candidates, then a sentence
   transformer re-ranks the top subset; the best-matching observed state is
   designated the *sub-goal*.

4. **Sub-goal protocol** – :func:`build_sequential_instruction_following_protocol`
   wraps the matched sub-goals in a
   :class:`SequentialInstructionFollowingProtocol` that adds a shaped-reward
   bonus whenever the agent reaches each sub-goal state during RL training.

Quick start
-----------
::

    from rlip.instruction_following import build_sequential_instruction_following_protocol
    from rlip.environments.registry import registry

    env = registry.get("Sailing-v0").create()

    protocol = build_sequential_instruction_following_protocol(
        instructions=["sail towards the beach side"],
        env=env,
        seed=0,
    )
    result = protocol(env)
    print(result)

Using a Claude/OpenAI agent for exploration
-------------------------------------------
Pass any protocol instance — including one whose observations come from an
agent-driven session — as ``exploration_protocol``::

    from rlip.interaction_protocols import GreedyEpisodeProtocol
    from rlip.adapters.openai_agent import RLIPAgent

    # Let the Claude agent explore and record a trajectory.
    agent = RLIPAgent(base_url="http://localhost:11434/v1", model="llama3.1")
    # (agent drives the env externally; collect observations into a protocol)

    greedy = GreedyEpisodeProtocol(
        policy_fn=lambda obs: 0,   # replace with agent-backed policy
        max_steps=100,
        record_history=True,
        translate=True,
    )
    protocol = build_sequential_instruction_following_protocol(
        instructions=["sail close to the harbor edge"],
        env=env,
        exploration_protocol=greedy,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .instruction_matching import BaseEncoder, TextEncoder, TFIDFEncoder
from .instruction_matching.matcher import DEFAULT_REFINE_TOP_K
from .instruction_matching.feedback import FeedbackLayer, get_feedback_layer
from .instruction_matching.matcher import score_instruction_against_corpus
from .interaction_protocols import (
    EpisodeResult,
    InstructionFollowingProtocol,
    InteractionResult,
    RandomEpisodeProtocol,
    StepRecord,
    _BaseProtocol,
    _EnvLike,
    _TranslateArg,
    _get,
    _get as _get_field,
    _make_sampler,
)
from .language_translation import get_translator
from .language_translation.base import LanguageTranslator
from .language_translation.caching import (
    CachingTranslator,
    clear_translation_cache,
    translation_cache_info,
)


# ── Reward-range utilities ────────────────────────────────────────────────────

def infer_max_reward(env: Any) -> float:
    """
    Infer the maximum achievable per-step reward for *env*.

    Walks the wrapper chain (up to 5 levels) looking for a ``reward_range``
    attribute — the Gymnasium convention is a ``(min, max)`` tuple.  Returns
    the upper bound when it is finite and positive.

    Falls back to ``1.0`` for native RLIP environments (e.g. Sailing-v0,
    GridWorld) where the goal reward is +1 by convention.

    Parameters
    ----------
    env:
        Any RLIP or Gymnasium environment (possibly wrapped).

    Returns
    -------
    float
        Estimated maximum positive reward.
    """
    candidate = env
    for _ in range(5):
        rr = getattr(candidate, "reward_range", None)
        if rr is not None:
            try:
                hi = float(rr[1])
                if math.isfinite(hi) and hi > 0:
                    return hi
            except (TypeError, IndexError):
                pass
        next_env = getattr(candidate, "_env", None) or getattr(candidate, "_wrapper", None)
        if next_env is None or next_env is candidate:
            break
        candidate = next_env
    return 1.0  # safe default for RLIP native envs (+1 goal reward)


def scale_sub_goal_bonus(env: Any, n_instructions: int = 1) -> float:
    """
    Compute an automatically scaled sub-goal bonus reward.

    Formula::

        bonus = max_reward / (100 × n_instructions)

    This ensures the sub-goal shaping signal is always two orders of
    magnitude smaller than the environment's actual goal reward, and
    shrinks further when many sub-goal descriptions are simultaneously
    active (so the total shaped reward never dwarfs the goal reward).

    Parameters
    ----------
    env:
        The RLIP environment being used (or any wrapper around one).
        Its reward range is probed via :func:`infer_max_reward`.
    n_instructions:
        Total number of active sub-goal language descriptions.
        Pass ``len(matched_states)`` or the count of simultaneous
        instructions to share the budget across all of them.

    Returns
    -------
    float
        Scaled bonus reward (always > 0).

    Example
    -------
    ::

        # Sailing-v0: max_reward=1.0, 3 sub-goals → bonus=0.00333...
        bonus = scale_sub_goal_bonus(env, n_instructions=3)
    """
    max_r = infer_max_reward(env)
    return max_r / (100.0 * max(1, n_instructions))


# ── Observation state cache ───────────────────────────────────────────────────
# Keyed by env_id → { language_description → raw_observation }
# Populated during exploration; reused by subsequent match_instruction calls
# for the same environment so that re-exploration is skipped.
_OBS_CACHE: dict[str, dict[str, Any]] = {}

# ── Instruction cache ─────────────────────────────────────────────────────────
# Keyed by env_id → { instruction_text → InstructionCacheEntry }
# Populated by match_instruction; success stats updated by
# InstructionFollowingProtocol.__call__ after each episode.
_INSTRUCTION_CACHE: "dict[str, dict[str, InstructionCacheEntry]]" = {}

# ── State → instruction mapping ───────────────────────────────────────────────
# Keyed by env_id → { language_description → [instruction, ...] }
# Records every instruction matched to each observed state so that a step
# visiting that state can be annotated even when a different instruction is
# currently active.
_STATE_INSTRUCTIONS: dict[str, dict[str, list[str]]] = {}


def clear_obs_cache(env_id: str | None = None) -> None:
    """
    Remove entries from the observation state cache.

    Parameters
    ----------
    env_id:
        Clear only entries for this environment.  When *None* the entire
        observation cache is cleared (all environments).

    Example
    -------
    ::

        from rlip.instruction_following import clear_obs_cache
        clear_obs_cache("Sailing-v0")   # clear one env
        clear_obs_cache()               # clear all
    """
    if env_id is None:
        _OBS_CACHE.clear()
    else:
        _OBS_CACHE.pop(env_id, None)


def obs_cache_info(env_id: str | None = None) -> dict[str, int]:
    """
    Return the number of cached observed states per environment.

    Parameters
    ----------
    env_id:
        Restrict to a single environment.  When *None* all environments
        are included.

    Returns
    -------
    dict[str, int]
        Mapping of ``env_id → number_of_unique_observed_states``.
    """
    if env_id is not None:
        return {env_id: len(_OBS_CACHE.get(env_id, {}))}
    return {k: len(v) for k, v in _OBS_CACHE.items()}


def obs_cache_langs(env_id: str) -> list[str]:
    """
    Return all language observations currently cached for *env_id*.

    This is a snapshot of the corpus built during the most recent call to
    :func:`match_instruction` or :func:`build_sequential_instruction_following_protocol`
    for the given environment.  The list is empty if no exploration has
    been run yet.

    Parameters
    ----------
    env_id:
        Registered environment ID, e.g. ``"Sailing-v0"``.

    Returns
    -------
    list[str]
        Unique language descriptions observed in the environment.

    Example
    -------
    ::

        from rlip.instruction_following import obs_cache_langs
        langs = obs_cache_langs("Sailing-v0")
        print(langs[:5])
    """
    return list(_OBS_CACHE.get(env_id, {}).keys())


# ── Match result dataclass ────────────────────────────────────────────────────

@dataclass
class InstructionMatch:
    """Outcome of matching an instruction against a set of observed states."""

    instruction: str
    """The original natural-language instruction."""

    matched_language: str
    """Observed language description with the highest similarity to *instruction*."""

    matched_observation: Any
    """Raw environment observation corresponding to *matched_language*."""

    similarity_score: float
    """Cosine similarity between *instruction* and *matched_language* (0–1)."""

    matched_states: list[tuple[str, Any, float]] = field(default_factory=list, repr=False)
    """
    All ``(language, observation, score)`` triples whose similarity to the
    instruction falls within *similarity_band* of the best score (inclusive).
    Sorted descending by score.  Always contains at least the primary match.
    These are used as the multi-state sub-goal set for reward shaping.
    """

    all_scores: list[tuple[str, float]] = field(default_factory=list, repr=False)
    """All ``(language, score)`` pairs, sorted descending by score."""

    def __str__(self) -> str:
        return (
            f"InstructionMatch(\n"
            f"  instruction:   {self.instruction!r}\n"
            f"  matched:       {self.matched_language!r}\n"
            f"  similarity:    {self.similarity_score:.4f}\n"
            f"  sub-goals:     {len(self.matched_states)} state(s)\n"
            f")"
        )


@dataclass
class InstructionCacheEntry:
    """Cached instruction match with accumulated episode success statistics."""

    env_id: str
    """Environment this instruction was matched against."""

    instruction: str
    """The original natural-language instruction text."""

    match: "InstructionMatch"
    """The most-recent :class:`InstructionMatch` result for this instruction."""

    episodes_run: int = 0
    """Total number of :class:`InstructionFollowingProtocol` episodes recorded."""

    successes: int = 0
    """Episodes in which the sub-goal was reached at least once."""

    @property
    def success_rate(self) -> float:
        """Fraction of completed episodes in which the sub-goal was reached."""
        return self.successes / self.episodes_run if self.episodes_run > 0 else 0.0

    def record_episode(self, sub_goal_reached: bool) -> None:
        """Called by :class:`InstructionFollowingProtocol` after each episode."""
        self.episodes_run += 1
        if sub_goal_reached:
            self.successes += 1

    def __str__(self) -> str:
        rate = f"{self.success_rate:.1%}" if self.episodes_run > 0 else "n/a"
        return (
            f"InstructionCacheEntry(\n"
            f"  instruction:  {self.instruction!r}\n"
            f"  matched:      {self.match.matched_language!r}\n"
            f"  similarity:   {self.match.similarity_score:.4f}\n"
            f"  episodes:     {self.episodes_run}  "
            f"successes={self.successes}  rate={rate}\n"
            f")"
        )


# ── Instruction planning database ─────────────────────────────────────────────
# Moved to instruction_plan_db.py; re-exported here for backward compatibility.

from .instruction_plan_db import (  # noqa: E402
    InstructionUsageRecord,
    InstructionPlanEntry,
    InstructionPlanDatabase,
    _PLAN_DATABASES,
    get_plan_database,
)


# ── Core matching logic ───────────────────────────────────────────────────────

def match_instruction(
    instruction: str,
    env: _EnvLike,
    *,
    encoder: Optional[BaseEncoder] = None,
    translator: Optional[LanguageTranslator] = None,
    exploration_protocol: Optional[_BaseProtocol] = None,
    max_steps: int = 200,
    seed: Optional[int] = None,
    similarity_band: float = 0.05,
    use_raw_observations: bool = False,
    feedback_layer: Optional[FeedbackLayer] = None,
    use_feedback: bool = True,
    refine_top_k: int = DEFAULT_REFINE_TOP_K,
) -> InstructionMatch:
    """
    Explore *env* and find the observed state whose language description
    best matches *instruction* using two-stage scoring.

    Pipeline
    --------
    1. Run *exploration_protocol* (default: random episode) to collect a
       trajectory with language annotations.
    2. Deduplicate identical language descriptions, keeping the first
       corresponding raw observation.
    3. Rank all candidates with TF-IDF cosine similarity and keep the top
       *refine_top_k*.
    4. Re-score that subset with a sentence transformer (or *encoder* when
       provided) and apply optional feedback adjustments.
    5. Return the state with the highest adjusted similarity as an
       :class:`InstructionMatch`.

    Parameters
    ----------
    instruction:
        Natural-language goal, e.g. ``"sail towards the beach side"``.
    env:
        An RLIP environment instance.  The exploration protocol resets it
        internally.
    encoder:
        Stage-2 refine encoder.  Must satisfy the
        :class:`~rlip.instruction_matching.BaseEncoder` interface.
        Defaults to sentence-transformers (``all-MiniLM-L6-v2``) when *None*.
        Pass :class:`~rlip.instruction_matching.TFIDFEncoder` to disable
        semantic re-ranking and use TF-IDF only.
    refine_top_k:
        Number of TF-IDF top candidates to re-score with the refine encoder.
    translator:
        :class:`~rlip.language_translation.LanguageTranslator` to convert
        raw observations to text.  Auto-resolved from the environment's
        ``env_id`` when *None*.
    exploration_protocol:
        Protocol used to collect observations.  Must be configured with
        ``record_history=True``.  Defaults to
        :class:`~rlip.interaction_protocols.RandomEpisodeProtocol` with
        *max_steps*, *seed*, and the resolved translator.
    max_steps:
        Step budget for the default random-exploration protocol.
    seed:
        Reproducibility seed for the default exploration protocol.
    similarity_band:
        States whose cosine similarity is within *similarity_band* of the
        best score (i.e. ``score >= best_score - similarity_band``) are
        collected into :attr:`InstructionMatch.matched_states` and treated
        as equivalent sub-goals for reward shaping.  The default of ``0.05``
        captures near-identical states.  Use ``0.0`` to restrict to the
        single best state only.
    use_raw_observations:
        When *False* (default) a language translator is **required** — the
        system always converts raw observations to natural-language strings
        before encoding and comparing them to the instruction.  Pass
        *True* only when the environment's observations are already
        human-readable strings (e.g. a text-adventure game) **or** when
        the LLM/user explicitly requests raw-observation matching.  In
        that case, if no translator is provided, ``str(obs)`` is used as
        the language representation.
    feedback_layer:
        Optional :class:`~rlip.instruction_matching.FeedbackLayer` used to
        adjust cosine scores from validated match feedback.  When *None*
        and *use_feedback* is True, loads the persisted layer for *env_id*.
    use_feedback:
        When True (default), apply validated feedback adjustments during
        scoring.  Set False to force raw TF-IDF cosine similarity only.

    Returns
    -------
    InstructionMatch

    Raises
    ------
    ValueError
        If *use_raw_observations* is ``False`` (the default) and no
        translator is available for the environment.
    RuntimeError
        If the exploration episode produces no observations.
    """
    env_id = getattr(env, "env_id", type(env).__name__)

    # ── Resolve translator ─────────────────────────────────────────────────────
    if translator is None:
        translator = get_translator(env_id)
    if translator is None:
        if not use_raw_observations:
            raise ValueError(
                f"No language translator registered for environment '{env_id}'. "
                "Pass an explicit translator= argument, register one in "
                "rlip.language_translation.TRANSLATORS, or set "
                "use_raw_observations=True to match against raw observation "
                "strings directly."
            )
        # Explicit raw-observation mode: wrap str() as a minimal translator so
        # the rest of the pipeline (exploration, caching, encoding) is uniform.
        class _RawObsTranslator(LanguageTranslator):
            name = "raw_obs"
            def translate(self, state: Any, **_: Any) -> str:  # type: ignore[override]
                return str(state)
        translator = _RawObsTranslator()

    # Wrap the translator with a caching layer so every translate() call is
    # memoised across this and all future match_instruction calls for env_id.
    if not isinstance(translator, CachingTranslator):
        caching_translator: LanguageTranslator = CachingTranslator(translator, env_id)
    else:
        caching_translator = translator

    # ── Observation cache lookup ───────────────────────────────────────────────
    # _OBS_CACHE[env_id] is a {language → raw_obs} dict built from all prior
    # explorations.  If it is non-empty we use it as our starting corpus and
    # skip running a new exploration episode.  A caller-provided
    # exploration_protocol always runs regardless (user opt-in to fresh data).
    cached_seen: dict[str, Any] = _OBS_CACHE.get(env_id, {})

    if cached_seen and exploration_protocol is None:
        # Cache hit for the default exploration path — skip re-exploration.
        seen: dict[str, Any] = dict(cached_seen)
    else:
        # ── Exploration ───────────────────────────────────────────────────────
        # Start from cached observations so new exploration only *adds* states.
        seen = dict(cached_seen)

        if exploration_protocol is not None:
            proto: _BaseProtocol = exploration_protocol
        else:
            from .interaction_protocols import MultiEpisodeProtocol
            _ep_len = 10  # conservative per-episode budget for short-episode envs
            _n_ep = max(1, max_steps // _ep_len)
            proto = MultiEpisodeProtocol(
                RandomEpisodeProtocol(
                    max_steps=_ep_len,
                    seed=seed,
                    record_history=True,
                    translate=caching_translator,
                ),
                n_episodes=_n_ep,
                base_seed=seed,
            )
        exploration_result = proto(env)

        # Collect (raw_obs, language_obs) from every episode step.
        new_pairs: list[tuple[Any, str]] = []
        for ep in exploration_result.episodes:
            for rec in ep.history:
                lang = rec.language_obs
                if lang is None:
                    # Fallback: translate inline (caching translator memoises).
                    lang = caching_translator.translate(rec.observation)
                new_pairs.append((rec.observation, lang))

        # Merge new observations into seen (language → obs, keep first per lang).
        for obs, lang in new_pairs:
            if lang not in seen:
                seen[lang] = obs

        if not seen:
            raise RuntimeError(
                "Exploration yielded no observations. "
                "Ensure the exploration protocol uses record_history=True."
            )

        # Persist combined set back to the obs cache for future calls.
        _OBS_CACHE[env_id] = seen

    candidates = list(seen.items())

    # ── Text encoding + similarity scoring ────────────────────────────────────
    _layer: FeedbackLayer | None = None
    if use_feedback:
        _layer = feedback_layer if feedback_layer is not None else get_feedback_layer(env_id)

    match_result = score_instruction_against_corpus(
        instruction,
        candidates,
        encoder=encoder,
        feedback_layer=_layer,
        similarity_band=similarity_band,
        refine_top_k=refine_top_k,
    )

    result_match = InstructionMatch(
        instruction=instruction,
        matched_language=match_result.best_language,
        matched_observation=match_result.best_observation,
        similarity_score=match_result.similarity_score,
        matched_states=match_result.matched_states,
        all_scores=match_result.all_scores,
    )

    # ── Update instruction cache ───────────────────────────────────────────────
    env_instr_cache = _INSTRUCTION_CACHE.setdefault(env_id, {})
    if instruction in env_instr_cache:
        # Preserve accumulated episode stats; refresh the match result.
        env_instr_cache[instruction].match = result_match
    else:
        env_instr_cache[instruction] = InstructionCacheEntry(
            env_id=env_id,
            instruction=instruction,
            match=result_match,
        )

    # ── Attach this instruction to each matched observed state ─────────────────
    state_map = _STATE_INSTRUCTIONS.setdefault(env_id, {})
    for lang, _obs, _sc in result_match.matched_states:
        instr_list = state_map.setdefault(lang, [])
        if instruction not in instr_list:
            instr_list.append(instruction)

    return result_match


# ── High-level builder ────────────────────────────────────────────────────────

def build_sequential_instruction_following_protocol(
    instructions: list[str],
    env: _EnvLike,
    *,
    encoder: Optional[BaseEncoder] = None,
    translator: Optional[LanguageTranslator] = None,
    exploration_protocol: Optional[_BaseProtocol] = None,
    policy_fn: Optional[Callable[[Any], Any]] = None,
    max_steps: int = 200,
    seed: Optional[int] = None,
    similarity_band: float = 0.05,
    sub_goal_bonus: Optional[float] = None,
    sub_goal_threshold: float = 0.5,
    record_history: bool = True,
    use_raw_observations: bool = False,
) -> "SequentialInstructionFollowingProtocol":
    """
    End-to-end builder for sequential multi-step instruction following.

    Takes a list of instructions that must be completed in order. The protocol:

    1. Explores and matches each instruction to its sub-goal state
    2. During training, gives reward only for the **current** active instruction
    3. When the current instruction is reached, moves to the next one
    4. Continues until all instructions are completed

    This allows decomposed multi-step goals to be learned sequentially with
    progressive sub-goal shaping.

    Parameters
    ----------
    instructions:
        List of natural-language instructions in order of completion.
        Example: ``["sail to the beach", "drop anchor", "return to harbor"]``
    env:
        RLIP environment to explore and train on.
    encoder:
        Text encoder for matching. Defaults to TFIDFEncoder when *None*.
    translator:
        Language translator. Auto-resolved from *env*'s ``env_id`` when *None*.
    exploration_protocol:
        Protocol used to collect exploration trajectory. Defaults to
        RandomEpisodeProtocol.
    policy_fn:
        Training policy ``Callable[[obs], action]``. Defaults to random sampling.
    max_steps:
        Step budget for exploration and training episodes.
    seed:
        Reproducibility seed.
    similarity_band:
        States within *similarity_band* of the best similarity score are
        treated as equivalent sub-goals.
    sub_goal_bonus:
        Reward bonus per instruction when sub-goal is reached.
        Auto-scaled via :func:`scale_sub_goal_bonus` when *None*.
    sub_goal_threshold:
        Cosine similarity threshold (0–1) to count as reaching a sub-goal.
    record_history:
        Whether to retain full step history in the result.
    use_raw_observations:
        Forwarded to :func:`match_instruction`.  When *False* (default) a
        language translator is required.  Set to *True* only when the
        environment observations are already text, or when the LLM/user
        explicitly requests raw-observation matching.

    Returns
    -------
    SequentialInstructionFollowingProtocol
        Ready-to-run protocol. Call with *env* to execute training.

    Example
    -------
    ::

        protocol = build_sequential_instruction_following_protocol(
            instructions=[
                "sail towards the beach",
                "approach the dock",
                "return to starting position"
            ],
            env=env,
            seed=42,
        )
        result = protocol(env)
        # Agent trained to complete all 3 instructions in sequence
    """
    if not instructions:
        raise ValueError("instructions list cannot be empty")

    # Match each instruction to find its sub-goal
    matches: list[InstructionMatch] = []
    env_id = getattr(env, "env_id", type(env).__name__)

    for instr in instructions:
        match = match_instruction(
            instr,
            env,
            encoder=encoder,
            translator=translator,
            exploration_protocol=exploration_protocol if matches == [] else None,
            max_steps=max_steps,
            seed=seed,
            similarity_band=similarity_band,
            use_raw_observations=use_raw_observations,
        )
        matches.append(match)

    # Retrieve cache entries for success tracking
    cache_entries = [
        _INSTRUCTION_CACHE.get(env_id, {}).get(instr)
        for instr in instructions
    ]

    state_instr_map = _STATE_INSTRUCTIONS.setdefault(env_id, {})
    translate_arg: _TranslateArg = translator if translator is not None else True

    # Auto-scale bonus distributed across all instructions
    n_sub_goals = sum(len(m.matched_states) for m in matches)
    effective_bonus: float = (
        sub_goal_bonus
        if sub_goal_bonus is not None
        else scale_sub_goal_bonus(env, n_instructions=len(instructions))
    )

    return SequentialInstructionFollowingProtocol(
        instructions=instructions,
        matches=matches,
        policy_fn=policy_fn,
        sub_goal_bonus=effective_bonus,
        sub_goal_threshold=sub_goal_threshold,
        max_steps=max_steps,
        seed=seed,
        record_history=record_history,
        translate=translate_arg,
        _stats_trackers=cache_entries,
        _state_instruction_map=state_instr_map,
    )


# ── Sequential instruction-following protocol ──────────────────────────────────

class SequentialInstructionFollowingProtocol(_BaseProtocol):
    """
    Episode protocol with sequential multi-step instruction following.

    Takes an ordered list of instructions and adds reward shaping for each
    one. The protocol:

    1. At each step, computes similarity to the **current** active instruction
    2. Gives a bonus reward when similarity threshold is met
    3. Upon reaching the current instruction, advances to the next one
    4. Continues until all instructions are completed

    This enables decomposed multi-step goals to be learned with progressive
    sub-goal shaping.

    Parameters
    ----------
    instructions:
        List of natural-language instructions in order.
    matches:
        Corresponding list of :class:`InstructionMatch` objects (one per
        instruction), each containing sub-goal language descriptions and
        observations.
    policy_fn:
        Policy ``Callable[[obs], action]``. Defaults to random sampling.
    sub_goal_bonus:
        Bonus reward per instruction when similarity threshold is met.
    sub_goal_threshold:
        Cosine similarity threshold (0–1) required to reach sub-goal.
    max_steps:
        Hard cap on episode length.
    seed:
        Seed for environment reset and random action sampling.
    record_history:
        Store full step trajectory in the result.
    translate:
        Translation source. Defaults to *True* (auto-lookup).
    """

    name = "sequential_instruction_following"

    def __init__(
        self,
        instructions: list[str],
        matches: list[InstructionMatch],
        policy_fn: Optional[Callable[[Any], Any]] = None,
        sub_goal_bonus: float = 0.1,
        sub_goal_threshold: float = 0.5,
        max_steps: int = 200,
        seed: Optional[int] = None,
        record_history: bool = True,
        translate: _TranslateArg = True,
        encoder_factory: Optional[Any] = None,
        _stats_trackers: Optional[list[Any]] = None,
        _state_instruction_map: Optional[dict] = None,
    ) -> None:
        if len(instructions) != len(matches):
            raise ValueError("instructions and matches must have the same length")
        
        self.instructions = instructions
        self.matches = matches
        self.policy_fn = policy_fn
        self.sub_goal_bonus = sub_goal_bonus
        self.sub_goal_threshold = sub_goal_threshold
        self.max_steps = max_steps
        self.seed = seed
        self.record_history = record_history
        self.translate = translate
        self._encoder_factory = encoder_factory
        self._stats_trackers = _stats_trackers or [None] * len(instructions)
        self._state_instruction_map = _state_instruction_map
        
        self._encoders: list[Any] = []
        self._sub_goal_vecs: list[list[Any]] = []

    def __call__(self, env: _EnvLike) -> InteractionResult:
        from .instruction_matching import TextEncoder  # noqa: PLC0415

        env_id = self._env_id(env)
        result = InteractionResult(protocol_name=self.name, env_id=env_id)
        translator = self._get_effective_translator(self.translate, env_id, None, 0, "")

        # Build encoders for each instruction if not already built
        if not self._encoders:
            factory = self._encoder_factory if self._encoder_factory is not None else TextEncoder
            for match in self.matches:
                corpus = [match.instruction] + [lg for lg, _, _ in match.matched_states]
                encoder = factory().fit(corpus)
                self._encoders.append(encoder)
                vecs = [encoder.encode(lg) for lg, _, _ in match.matched_states]
                self._sub_goal_vecs.append(vecs)

        reset_out = env.reset(seed=self.seed)
        obs = _get(reset_out, "observation", reset_out)

        sampler = _make_sampler(env, self.seed)
        policy = self.policy_fn if self.policy_fn is not None else (lambda _obs: sampler())

        total_reward = 0.0
        history: list[StepRecord] = []
        action_history: list[Any] = []
        
        # Track which instruction is currently active (0 = first, etc.)
        current_instr_idx = 0
        instr_reached_flags = [False] * len(self.instructions)

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

            # ── Language-based sequential sub-goal shaping ────────────────────
            language_obs: Optional[str] = None
            if translator:
                language_obs = translator.translate(obs, action_history=action_history)

                # Annotate steps with associated instructions
                if self._state_instruction_map and language_obs in self._state_instruction_map:
                    associated = self._state_instruction_map[language_obs]
                    if associated:
                        info["associated_instructions"] = list(associated)

                # Only check similarity for the current active instruction
                if current_instr_idx < len(self.instructions):
                    current_match = self.matches[current_instr_idx]
                    encoder = self._encoders[current_instr_idx]
                    obs_vec = encoder.encode(language_obs)
                    
                    # Max similarity across all states for current instruction
                    sim = max(
                        encoder.cosine_similarity(obs_vec, sg_vec)
                        for sg_vec in self._sub_goal_vecs[current_instr_idx]
                    )
                    info["sub_goal_similarity"] = round(sim, 4)
                    info["current_instruction"] = current_instr_idx
                    info["total_instructions"] = len(self.instructions)

                    if sim >= self.sub_goal_threshold and not instr_reached_flags[current_instr_idx]:
                        # Reached current instruction — give bonus and advance
                        reward += self.sub_goal_bonus
                        instr_reached_flags[current_instr_idx] = True
                        info["sub_goal_reached"] = True
                        info["instruction_reached"] = self.instructions[current_instr_idx]
                        
                        # Move to next instruction if available
                        if current_instr_idx + 1 < len(self.instructions):
                            current_instr_idx += 1

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

        # Record final success metrics for each instruction
        n_complete = sum(instr_reached_flags)
        result.metadata = {
            "instructions_completed": n_complete,
            "total_instructions": len(self.instructions),
            "completion_rate": n_complete / len(self.instructions),
        }
        
        # Update stats trackers for each instruction
        for idx, tracker in enumerate(self._stats_trackers):
            if tracker is not None:
                tracker.record_episode(instr_reached_flags[idx])

        end_reason = (
            "terminated" if terminated
            else "truncated" if truncated
            else "max_steps"
        )

        result.episodes.append(
            EpisodeResult(
                episode=len(result.episodes),
                steps=step_n,
                total_reward=total_reward,
                terminated=terminated,
                history=history,
                metadata={
                    "end_reason": end_reason,
                    "instructions_reached": sum(instr_reached_flags),
                    "instructions_completed": n_complete,
                },
            )
        )

        return result


# ── Language-tracking environment wrapper ────────────────────────────────────

class LanguageTrackingWrapper:
    """
    Transparent wrapper around any RLIP environment that records
    language-described state visits during RL agent training.

    Pass this wrapper in place of the raw environment to any agent's
    ``.train()`` call.  After training is complete, pass the wrapper to
    :func:`derive_instructions_from_training` (or use the all-in-one
    :func:`train_and_derive_instructions` helper) to extract instruction
    candidates.

    Parameters
    ----------
    env:
        The underlying RLIP environment.
    translator:
        :class:`~rlip.language_translation.LanguageTranslator` used to
        convert each raw observation to a language description.
        Auto-resolved from the environment's ``env_id`` when *None*.
        If no translator can be resolved the wrapper still functions but
        accumulates no language data (and
        :func:`derive_instructions_from_training` will return an empty list).
    success_fn:
        ``Callable[[obs, reward, terminated, truncated, info], bool]`` that
        decides whether the current step concludes a *successful* episode.
        Defaults to ``terminated is True``, which is the standard convention
        for goal-reaching environments in RLIP (e.g. Sailing-v0, GridWorld).
        Override this for environments where success is reward-based or where
        ``terminated`` fires on failure instead of success.
    """

    def __init__(
        self,
        env: _EnvLike,
        translator: Optional[LanguageTranslator] = None,
        success_fn: Optional[Callable[..., bool]] = None,
    ) -> None:
        self._env = env
        self._env_id: str = getattr(env, "env_id", type(env).__name__)

        # Wrap with a caching layer so translations are memoised globally.
        if translator is None:
            translator = get_translator(self._env_id)
        if translator is not None and not isinstance(translator, CachingTranslator):
            translator = CachingTranslator(translator, self._env_id)
        self._translator: Optional[LanguageTranslator] = translator

        self._success_fn: Callable[..., bool] = (
            success_fn if success_fn is not None
            else lambda obs, reward, term, trunc, info: bool(term)
        )

        # Episode-level state (reset on each env.reset() call).
        self._ep_idx: int = -1
        self._current_ep_langs: list[str] = []
        self._action_history: list[Any] = []

        # Aggregate tracking across all episodes.
        # lang → set of episode indices in which this state was visited.
        self._lang_episodes: dict[str, set[int]] = {}
        # lang → set of episode indices that were *successes* and visited this state.
        self._lang_success_episodes: dict[str, set[int]] = {}
        # lang → one representative raw observation (used to populate _OBS_CACHE).
        self._lang_obs_sample: dict[str, Any] = {}

    # ── env interface ─────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Any:
        self._ep_idx += 1
        self._current_ep_langs = []
        self._action_history = []
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:
        self._action_history.append(action)
        result = self._env.step(action)

        if self._translator is not None:
            obs    = _get_field(result, "observation", result)
            term   = bool(_get_field(result, "terminated", False))
            trunc  = bool(_get_field(result, "truncated", False))
            reward = float(_get_field(result, "reward", 0.0))
            info   = _get_field(result, "info", {}) or {}

            lang = self._translator.translate(obs, action_history=list(self._action_history))
            self._current_ep_langs.append(lang)
            self._lang_episodes.setdefault(lang, set()).add(self._ep_idx)
            if lang not in self._lang_obs_sample:
                self._lang_obs_sample[lang] = obs

            if self._success_fn(obs, reward, term, trunc, info):
                # Mark every unique state visited this episode as part of a success.
                for s in set(self._current_ep_langs):
                    self._lang_success_episodes.setdefault(s, set()).add(self._ep_idx)

        return result

    def close(self) -> None:
        return self._env.close()

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def env_id(self) -> str:
        return self._env_id

    def __getattr__(self, name: str) -> Any:
        """Transparent pass-through for any attribute not defined on the wrapper."""
        return getattr(self._env, name)

    def __repr__(self) -> str:
        return f"LanguageTrackingWrapper({self._env!r})"


# ── Post-training instruction derivation ──────────────────────────────────────

def derive_instructions_from_training(
    wrapper: LanguageTrackingWrapper,
    *,
    top_k: int = 5,
    min_episode_visits: int = 2,
    baseline_success_rate: Optional[float] = None,
    long_term_goal: Optional[str] = None,
    llm_state_planner: Optional[Callable[[str, str, list[str]], list[str]]] = None,
) -> list[InstructionCacheEntry]:
    """
    Analyse the language state visit log collected by *wrapper* during RL
    training and derive up to *top_k* instruction strings most correlated
    with successful episodes.

    Scoring
    -------
    Each unique language state *s* observed during training is scored by::

        score(s) = csr(s) × log₂(1 + episode_visits(s))

    where:

    - ``csr(s)`` is the *conditional success rate*: the fraction of episodes
      that both visited *s* and ended successfully (``terminated=True`` by
      default, overridable via ``success_fn`` on the wrapper).
    - ``episode_visits(s)`` is the number of distinct training episodes in
      which *s* appeared.  The log term privileges frequently-visited states.

    States appearing in fewer than *min_episode_visits* episodes are ignored.
    When *baseline_success_rate* is supplied, only states that strictly
    exceed it are included (useful to filter states no better than the
    overall per-episode success rate).

    Each selected language description is treated as its own instruction
    (the description *is* the goal to reach) and is registered in
    ``_INSTRUCTION_CACHE`` and ``_STATE_INSTRUCTIONS`` so it can be
    retrieved by :func:`list_instruction_cache` and passed directly to
    :func:`build_sequential_instruction_following_protocol`.

    Parameters
    ----------
    wrapper:
        A :class:`LanguageTrackingWrapper` whose ``.step()`` has been called
        (i.e. training has already run).
    top_k:
        Maximum number of instruction candidates to derive.
    min_episode_visits:
        States visited in fewer than this many distinct episodes are ignored.
    baseline_success_rate:
        Optional lower-bound filter on conditional success rate.
    long_term_goal:
        Optional high-level RL objective string.  When provided alongside
        *llm_state_planner*, this goal is used to ask the LLM for an ordered
        state progression toward the objective.
    llm_state_planner:
        Optional callback used to produce a milestone instruction sequence from
        observed successful states.  Signature:
        ``(env_id, long_term_goal, candidate_states) -> list[str]``.
        Each returned instruction is matched against cached observed states and
        registered in the instruction cache.

    Returns
    -------
    list[InstructionCacheEntry]
        Derived cache entries sorted best-first by score.  An empty list is
        returned when the wrapper collected no language data (no translator).
    """
    env_id = wrapper._env_id

    # Merge observed states into the global observation cache so they are
    # available for subsequent match_instruction calls.
    obs_cache = _OBS_CACHE.setdefault(env_id, {})
    for lang, obs in wrapper._lang_obs_sample.items():
        if lang not in obs_cache:
            obs_cache[lang] = obs

    # Score each candidate language state.
    scored: list[tuple[str, float, float, int, int]] = []
    for lang, ep_set in wrapper._lang_episodes.items():
        n_eps = len(ep_set)
        if n_eps < min_episode_visits:
            continue
        n_success = len(wrapper._lang_success_episodes.get(lang, set()))
        csr = n_success / n_eps
        if baseline_success_rate is not None and csr <= baseline_success_rate:
            continue
        score = csr * math.log2(1.0 + n_eps)
        scored.append((lang, score, csr, n_eps, n_success))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    # Optional LLM-guided state progression planning:
    # turn the strongest successful-state candidates into an ordered milestone set.
    fallback_instructions: list[str] = [lang for lang, *_rest in top]
    selected_instructions: list[str] = list(fallback_instructions)
    if llm_state_planner is not None and top:
        goal_text = (long_term_goal or "").strip()
        if goal_text:
            candidate_states = [lang for lang, *_rest in scored[: max(top_k * 3, top_k)]]
            try:
                llm_steps = llm_state_planner(env_id, goal_text, candidate_states)
            except Exception:
                llm_steps = []

            normalized_steps: list[str] = []
            seen_steps: set[str] = set()
            for step in llm_steps:
                cleaned = " ".join(str(step).split()).strip(" .")
                key = cleaned.lower()
                if not cleaned or len(cleaned) < 3 or key in seen_steps:
                    continue
                seen_steps.add(key)
                normalized_steps.append(cleaned)
                if len(normalized_steps) >= top_k:
                    break

            if normalized_steps:
                selected_instructions = normalized_steps

    instr_cache = _INSTRUCTION_CACHE.setdefault(env_id, {})
    state_map   = _STATE_INSTRUCTIONS.setdefault(env_id, {})
    entries: list[InstructionCacheEntry] = []

    for instruction in selected_instructions:
        # Fast path for exact observed language states.
        if instruction in wrapper._lang_obs_sample:
            raw_obs = wrapper._lang_obs_sample.get(instruction)
            im = InstructionMatch(
                instruction=instruction,
                matched_language=instruction,
                matched_observation=raw_obs,
                similarity_score=1.0,
                matched_states=[(instruction, raw_obs, 1.0)],
                all_scores=[(instruction, 1.0)],
            )

            if instruction in instr_cache:
                # Preserve existing episode stats; refresh the match.
                instr_cache[instruction].match = im
                entry = instr_cache[instruction]
            else:
                entry = InstructionCacheEntry(env_id=env_id, instruction=instruction, match=im)
                instr_cache[instruction] = entry

            instr_list = state_map.setdefault(instruction, [])
            if instruction not in instr_list:
                instr_list.append(instruction)
            entries.append(entry)
            continue

        # LLM-proposed milestones are mapped back to concrete observed states.
        # Always prefer translated language-state matching when a translator is
        # available on the wrapper.
        translator_available = wrapper._translator is not None
        try:
            mapped = match_instruction(
                instruction,
                wrapper,
                translator=wrapper._translator,
                exploration_protocol=None,
                use_raw_observations=not translator_available,
            )
            entry = instr_cache[instruction]
            entry.match = mapped
            entries.append(entry)
        except Exception:
            # Skip unmatched LLM output; deterministic fallback is applied below.
            continue

    # If all LLM-proposed milestones failed to map, fall back to top scored states.
    if not entries and selected_instructions != fallback_instructions:
        for lang in fallback_instructions:
            raw_obs = wrapper._lang_obs_sample.get(lang)
            if raw_obs is None:
                continue
            im = InstructionMatch(
                instruction=lang,
                matched_language=lang,
                matched_observation=raw_obs,
                similarity_score=1.0,
                matched_states=[(lang, raw_obs, 1.0)],
                all_scores=[(lang, 1.0)],
            )
            if lang in instr_cache:
                instr_cache[lang].match = im
                entry = instr_cache[lang]
            else:
                entry = InstructionCacheEntry(env_id=env_id, instruction=lang, match=im)
                instr_cache[lang] = entry
            instr_list = state_map.setdefault(lang, [])
            if lang not in instr_list:
                instr_list.append(lang)
            entries.append(entry)

    return entries


def train_and_derive_instructions(
    agent: Any,
    env: _EnvLike,
    n_episodes: int = 500,
    *,
    translator: Optional[LanguageTranslator] = None,
    success_fn: Optional[Callable[..., bool]] = None,
    top_k: int = 5,
    min_episode_visits: int = 2,
    baseline_success_rate: Optional[float] = None,
    long_term_goal: Optional[str] = None,
    llm_state_planner: Optional[Callable[[str, str, list[str]], list[str]]] = None,
    **train_kwargs: Any,
) -> tuple[Any, list[InstructionCacheEntry]]:
    """
    Train *agent* on *env* while tracking language state visits, then
    automatically derive and cache the top instruction candidates.

    This convenience function combines :class:`LanguageTrackingWrapper`,
    the agent's ``.train()`` method, and
    :func:`derive_instructions_from_training` in a single call.

    Parameters
    ----------
    agent:
        Any RLIP agent with a ``.train(env, n_episodes, **kwargs)`` method,
        e.g. :class:`~rlip.rl_agents.TabularQAgent`.
    env:
        The RLIP environment to train on.
    n_episodes:
        Number of training episodes to run.
    translator:
        Language translator.  Auto-resolved from *env*'s ``env_id`` when
        *None*.
    success_fn:
        Custom episode-success predicate passed to
        :class:`LanguageTrackingWrapper`.  Defaults to ``terminated=True``.
        Signature: ``(obs, reward, terminated, truncated, info) → bool``.
    top_k:
        Number of instruction candidates to derive after training.
    min_episode_visits:
        Minimum distinct episodes a state must appear in to be considered.
    baseline_success_rate:
        Optional filter: only states with a conditional success rate above
        this value will be included.
    long_term_goal:
        Optional high-level RL objective.  Used only when
        *llm_state_planner* is provided.
    llm_state_planner:
        Optional callback that asks an LLM to produce an ordered sequence of
        milestone instructions from successful observed states.
    **train_kwargs:
        Extra keyword arguments forwarded verbatim to ``agent.train()``.

    Returns
    -------
    tuple[TrainResult, list[InstructionCacheEntry]]
        ``(train_result, derived_instructions)`` where *train_result* is the
        object returned by ``agent.train()`` and *derived_instructions* are
        the cached :class:`InstructionCacheEntry` objects, sorted best-first.

    Example
    -------
    ::

        from rlip import TabularQAgent
        from rlip.instruction_following import (
            train_and_derive_instructions,
            list_instruction_cache,
        )
        from rlip.environments.registry import registry

        env = registry.get("Sailing-v0").create()
        agent = TabularQAgent()

        train_result, suggestions = train_and_derive_instructions(
            agent, env, n_episodes=500, top_k=5,
        )

        for entry in suggestions:
            print(entry)

        # Re-use a derived instruction for guided RL:
        from rlip.instruction_following import build_sequential_instruction_following_protocol
        best_instr = suggestions[0].instruction
        protocol = build_sequential_instruction_following_protocol([best_instr], env)
        result = protocol(env)
    """
    wrapper = LanguageTrackingWrapper(env, translator=translator, success_fn=success_fn)
    train_result = agent.train(wrapper, n_episodes=n_episodes, **train_kwargs)
    derived = derive_instructions_from_training(
        wrapper,
        top_k=top_k,
        min_episode_visits=min_episode_visits,
        baseline_success_rate=baseline_success_rate,
        long_term_goal=long_term_goal,
        llm_state_planner=llm_state_planner,
    )
    return train_result, derived


# ── Instruction-cache utilities ───────────────────────────────────────────────

def list_instruction_cache(
    env_id: str | None = None,
) -> dict[str, dict[str, "InstructionCacheEntry"]]:
    """
    Return the instruction cache, optionally filtered to one environment.

    Returns a shallow copy so callers cannot accidentally mutate the cache.

    Parameters
    ----------
    env_id:
        When given, restrict results to this environment.  When *None*
        all environments are included.

    Returns
    -------
    dict[str, dict[str, InstructionCacheEntry]]
        ``{env_id: {instruction_text: InstructionCacheEntry}}``

    Example
    -------
    ::

        from rlip.instruction_following import list_instruction_cache
        for env_id, entries in list_instruction_cache().items():
            for instr, entry in entries.items():
                print(env_id, entry)
    """
    if env_id is not None:
        return {env_id: dict(_INSTRUCTION_CACHE.get(env_id, {}))}
    return {k: dict(v) for k, v in _INSTRUCTION_CACHE.items()}


def clear_instruction_cache(env_id: str | None = None) -> None:
    """
    Remove entries from the instruction cache (and state→instruction map).

    Parameters
    ----------
    env_id:
        Clear only entries for this environment.  When *None* clears
        everything (all environments).
    """
    if env_id is None:
        _INSTRUCTION_CACHE.clear()
        _STATE_INSTRUCTIONS.clear()
    else:
        _INSTRUCTION_CACHE.pop(env_id, None)
        _STATE_INSTRUCTIONS.pop(env_id, None)


def state_instruction_map(
    env_id: str,
) -> dict[str, list[str]]:
    """
    Return the mapping of observed language descriptions → matched instructions
    for *env_id*.

    Each key is a unique language description that has been returned by the
    environment's translator and used as a sub-goal for at least one
    instruction.  The value is the (ordered) list of instruction strings that
    were matched to that state.

    Parameters
    ----------
    env_id:
        The environment identifier, e.g. ``"Sailing-v0"``.

    Returns
    -------
    dict[str, list[str]]
        A shallow copy of the live mapping.

    Example
    -------
    ::

        from rlip.instruction_following import state_instruction_map
        smap = state_instruction_map("Sailing-v0")
        for state, instrs in smap.items():
            print(state, "→", instrs)
    """
    return dict(_STATE_INSTRUCTIONS.get(env_id, {}))


__all__ = [
    "TextEncoder",
    "TFIDFEncoder",
    "InstructionMatch",
    "InstructionCacheEntry",
    "match_instruction",
    "build_sequential_instruction_following_protocol",
    "SequentialInstructionFollowingProtocol",
    # Reward scaling
    "infer_max_reward",
    "scale_sub_goal_bonus",
    # Language-tracking RL training
    "LanguageTrackingWrapper",
    "derive_instructions_from_training",
    "train_and_derive_instructions",
    # Instruction cache
    "_INSTRUCTION_CACHE",
    "list_instruction_cache",
    "clear_instruction_cache",
    # State → instruction map
    "_STATE_INSTRUCTIONS",
    "state_instruction_map",
    # Observation cache management
    "_OBS_CACHE",
    "clear_obs_cache",
    "obs_cache_info",
    "obs_cache_langs",
    "clear_translation_cache",
    "translation_cache_info",
]
