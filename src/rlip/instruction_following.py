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

3. **Instruction matching** – all descriptions and the instruction string are
   encoded as TF-IDF vectors (:class:`TextEncoder`); the observed state with
   the highest cosine similarity to the instruction is designated the
   *sub-goal*.

4. **Sub-goal protocol** – :func:`build_instruction_following_protocol`
   wraps the matched sub-goal in an
   :class:`~rlip.interaction_protocols.InstructionFollowingProtocol` that
   adds a shaped-reward bonus whenever the agent revisits that sub-goal during
   subsequent RL training.

Quick start
-----------
::

    from rlip.instruction_following import build_instruction_following_protocol
    from rlip.environments.registry import registry

    env = registry.get("Sailing-v0").create()

    protocol = build_instruction_following_protocol(
        instruction="sail towards the beach side",
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
    protocol = build_instruction_following_protocol(
        instruction="sail close to the harbor edge",
        env=env,
        exploration_protocol=greedy,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .instruction_matching import BaseEncoder, TextEncoder, TFIDFEncoder
from .interaction_protocols import (
    InstructionFollowingProtocol,
    RandomEpisodeProtocol,
    _BaseProtocol,
    _EnvLike,
    _TranslateArg,
)
from .language_translation import get_translator
from .language_translation.base import LanguageTranslator
from .language_translation.caching import (
    CachingTranslator,
    clear_translation_cache,
    translation_cache_info,
)

# ── Observation state cache ───────────────────────────────────────────────────
# Keyed by env_id → { language_description → raw_observation }
# Populated during exploration; reused by subsequent match_instruction calls
# for the same environment so that re-exploration is skipped.
_OBS_CACHE: dict[str, dict[str, Any]] = {}


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
) -> InstructionMatch:
    """
    Explore *env* and find the observed state whose language description
    best matches *instruction* using TF-IDF cosine similarity.

    Pipeline
    --------
    1. Run *exploration_protocol* (default: random episode) to collect a
       trajectory with language annotations.
    2. Deduplicate identical language descriptions, keeping the first
       corresponding raw observation.
    3. Fit a :class:`TextEncoder` on the corpus
       ``[instruction] + unique_language_descriptions``.
    4. Compute cosine similarity between the encoded instruction and each
       encoded language description.
    5. Return the state with the highest similarity as an
       :class:`InstructionMatch`.

    Parameters
    ----------
    instruction:
        Natural-language goal, e.g. ``"sail towards the beach side"``.
    env:
        An RLIP environment instance.  The exploration protocol resets it
        internally.
    encoder:
        Text encoder instance to use for similarity scoring.  Must satisfy
        the :class:`~rlip.instruction_matching.BaseEncoder` interface.
        Defaults to :class:`~rlip.instruction_matching.TFIDFEncoder` when
        *None*.  Pass a :class:`~rlip.instruction_matching.BM25Encoder` or
        :class:`~rlip.instruction_matching.SentenceEncoder` for improved
        matching quality.
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

    Returns
    -------
    InstructionMatch

    Raises
    ------
    ValueError
        If no translator is available for the environment and *translator*
        is not supplied.
    RuntimeError
        If the exploration episode produces no observations.
    """
    env_id = getattr(env, "env_id", type(env).__name__)

    # ── Resolve translator ─────────────────────────────────────────────────────
    if translator is None:
        translator = get_translator(env_id)
    if translator is None:
        raise ValueError(
            f"No language translator registered for environment '{env_id}'. "
            "Pass an explicit translator= argument or register one in "
            "rlip.language_translation.TRANSLATORS."
        )

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

        proto: _BaseProtocol = exploration_protocol or RandomEpisodeProtocol(
            max_steps=max_steps,
            seed=seed,
            record_history=True,
            translate=caching_translator,
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

    unique_langs = list(seen.keys())
    unique_obs = [seen[lg] for lg in unique_langs]

    # ── Text encoding + similarity scoring ────────────────────────────────────
    corpus = [instruction] + unique_langs
    _encoder: BaseEncoder = encoder if encoder is not None else TFIDFEncoder()
    _encoder.fit(corpus)
    instruction_vec = _encoder.encode(instruction)

    scored: list[tuple[str, Any, float]] = []
    for lang, obs in zip(unique_langs, unique_obs):
        obs_vec = _encoder.encode(lang)
        sim = _encoder.cosine_similarity(instruction_vec, obs_vec)
        scored.append((lang, obs, sim))

    scored.sort(key=lambda x: x[2], reverse=True)
    best_lang, best_obs, best_score = scored[0]
    all_scores = [(lg, sc) for lg, _, sc in scored]

    # Collect states within similarity_band of the best score as co-equal sub-goals.
    cutoff = best_score - similarity_band
    matched_states = [(lg, obs, sc) for lg, obs, sc in scored if sc >= cutoff]

    return InstructionMatch(
        instruction=instruction,
        matched_language=best_lang,
        matched_observation=best_obs,
        similarity_score=best_score,
        matched_states=matched_states,
        all_scores=all_scores,
    )


# ── High-level builder ────────────────────────────────────────────────────────

def build_instruction_following_protocol(
    instruction: str,
    env: _EnvLike,
    *,
    encoder: Optional[BaseEncoder] = None,
    translator: Optional[LanguageTranslator] = None,
    exploration_protocol: Optional[_BaseProtocol] = None,
    policy_fn: Optional[Callable[[Any], Any]] = None,
    max_steps: int = 200,
    seed: Optional[int] = None,
    similarity_band: float = 0.05,
    sub_goal_bonus: float = 1.0,
    sub_goal_threshold: float = 0.5,
    sub_goal_repeatable: bool = False,
    record_history: bool = True,
) -> InstructionFollowingProtocol:
    """
    End-to-end builder: explore, match, and return a ready-to-run
    :class:`~rlip.interaction_protocols.InstructionFollowingProtocol`.

    This is the primary entry-point for instruction-following RL.  It
    calls :func:`match_instruction` internally and wraps the result in
    the appropriate protocol.

    Parameters
    ----------
    instruction:
        Natural-language goal string, e.g.
        ``"sail towards the beach side"``.
    env:
        RLIP environment to explore and later train on.
    encoder:
        Text encoder for the exploration-phase matching step.  Defaults to
        :class:`~rlip.instruction_matching.TFIDFEncoder` when *None*.
    translator:
        Language translator.  Auto-resolved from *env*'s ``env_id`` when
        *None*.
    exploration_protocol:
        Protocol used to collect the exploration trajectory.  Defaults to
        :class:`~rlip.interaction_protocols.RandomEpisodeProtocol`.
        Can be an agent-driven protocol — any protocol that records
        ``language_obs`` in its step history works.
    policy_fn:
        Training policy ``Callable[[obs], action]``.  Defaults to random
        sampling when *None*.
    max_steps:
        Step budget shared between exploration and training episodes.
    seed:
        Reproducibility seed for the exploration and training resets.
    sub_goal_bonus:
        Reward bonus added when the sub-goal similarity threshold is met.
    sub_goal_threshold:
        Cosine similarity threshold (0–1) to count a step as reaching the
        sub-goal.  Tune this to the density of language descriptions in
        your environment.
    similarity_band:
        States within *similarity_band* of the best cosine similarity score
        are included as co-equal sub-goals (see
        :attr:`InstructionMatch.matched_states`).  The reward bonus fires
        whenever the agent reaches *any* of these states.  Default ``0.05``.
    sub_goal_repeatable:
        *False* (default) — **first-visit** semantics: the bonus fires once
        (the first step similarity >= *sub_goal_threshold*), then similarity
        is no longer computed for that episode so the agent has no incentive
        to linger at or return to the sub-goal.
        *True* — bonus fires on every step the threshold is met.
    record_history:
        Whether to retain the full step history in the training result.

    Returns
    -------
    InstructionFollowingProtocol
        Configured with the matched sub-goal; call it with *env* to run
        a training episode.

    Example
    -------
    ::

        protocol = build_instruction_following_protocol(
            instruction="sail towards the beach side",
            env=env,
            seed=42,
        )
        result = protocol(env)
        for rec in result.episodes[0].history:
            print(rec.step, rec.info.get("sub_goal_similarity"), rec.language_obs)
    """
    match = match_instruction(
        instruction,
        env,
        encoder=encoder,
        translator=translator,
        exploration_protocol=exploration_protocol,
        max_steps=max_steps,
        seed=seed,
        similarity_band=similarity_band,
    )

    # Use the resolved translator instance (or True for auto-lookup) so
    # InstructionFollowingProtocol doesn't have to re-resolve it.
    translate_arg: _TranslateArg = translator if translator is not None else True

    # Extract all co-equal sub-goal language descriptions from matched_states,
    # excluding the primary which is passed separately as sub_goal_language.
    extra_langs = [
        lg for lg, _obs, _sc in match.matched_states
        if lg != match.matched_language
    ]

    return InstructionFollowingProtocol(
        instruction=match.instruction,
        sub_goal_language=match.matched_language,
        sub_goal_observation=match.matched_observation,
        sub_goal_languages=extra_langs,
        policy_fn=policy_fn,
        sub_goal_bonus=sub_goal_bonus,
        sub_goal_threshold=sub_goal_threshold,
        sub_goal_repeatable=sub_goal_repeatable,
        max_steps=max_steps,
        seed=seed,
        record_history=record_history,
        translate=translate_arg,
    )


__all__ = [
    "TextEncoder",
    "TFIDFEncoder",
    "InstructionMatch",
    "match_instruction",
    "build_instruction_following_protocol",
    # Cache management
    "_OBS_CACHE",
    "clear_obs_cache",
    "obs_cache_info",
    "clear_translation_cache",
    "translation_cache_info",
]
