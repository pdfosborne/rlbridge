"""
Instruction-following MCP tools for the RLIP plugin.

Tools:
- rl_clear_obs_cache
- rl_match_instruction
- rl_instruction_run_episode
- rl_match_sequential_instructions
- rl_sequential_instruction_run_episode

Uses the shared ``_instruction_protocols`` cache from ``_state`` so that
rl_train_agent (in _tools_agents) can access previously-matched sub-goals.
"""

from __future__ import annotations

import asyncio
import re
import sys
from typing import Any, Optional

from mcp.server.fastmcp import Context

from ._prompts import decompose_instruction_vocab_prompt
from ._state import _instruction_protocols, mcp


# ---------------------------------------------------------------------------
# LLM sub-goal decomposition helper
# ---------------------------------------------------------------------------

def _extract_vocab_clauses(observed_langs: list[str]) -> list[str]:
    """
    Extract the recurring semantic phrases from a list of environment language
    descriptions using only general clause boundaries.

    Splits on commas, semicolons, and sentence terminals — no environment-
    specific keywords — so this works for any RLIP environment.  Returns only
    clauses that appear in at least two distinct descriptions (i.e. genuine
    vocabulary atoms, not one-off noise), sorted and deduplicated.
    """
    from collections import Counter
    clause_counts: Counter[str] = Counter()
    for desc in observed_langs:
        # Split on general separators: comma, semicolon, sentence boundary
        parts = re.split(r",\s+|;\s+|\.\s+", desc)
        seen_in_desc: set[str] = set()
        for part in parts:
            part = part.strip().rstrip(".")
            if 4 <= len(part) <= 100 and part not in seen_in_desc:
                clause_counts[part] += 1
                seen_in_desc.add(part)
    # Keep only clauses shared by two or more distinct descriptions
    return sorted(clause for clause, count in clause_counts.items() if count >= 2)


async def _decompose_instruction_with_llm(
    ctx: Context,
    instruction: str,
    env_id: str,
    observed_langs: list[str],
) -> list[str]:
    """
    Ask the host LLM to break *instruction* into clear, ordered sub-steps
    that use the environment's exact language vocabulary.

    Builds an explicit vocabulary lexicon from the translated language strings
    and requires the LLM to use those phrases verbatim rather than abstract
    domain jargon.

    Returns a list of sub-step strings, or an empty list on any failure.
    """
    import mcp.types as _t

    # Sample ~20 evenly-spaced translated language strings; keep full list for vocab extraction.
    n_sample = min(20, len(observed_langs))
    step = max(1, len(observed_langs) // n_sample)
    sample = observed_langs[::step][:n_sample]
    lang_block = "\n".join(f"  - {lg}" for lg in sample)
    if len(observed_langs) > n_sample:
        lang_block += f"\n  … ({len(observed_langs) - n_sample} more not shown)"

    # Extracted vocabulary atoms: the reusable clauses the LLM should copy.
    vocab_clauses = _extract_vocab_clauses(observed_langs)
    vocab_block = "\n".join(f"  • {v}" for v in vocab_clauses[:60])

    prompt = decompose_instruction_vocab_prompt(env_id, instruction, lang_block, vocab_block)

    try:
        result = await ctx.session.create_message(
            messages=[_t.SamplingMessage(
                role="user",
                content=_t.TextContent(type="text", text=prompt),
            )],
            max_tokens=400,
        )
    except Exception:
        return []

    # Extract text from result
    content = result.content
    if hasattr(content, "text"):
        raw = content.text
    elif isinstance(content, list) and content:
        raw = getattr(content[0], "text", "") or ""
    else:
        raw = str(content)

    # Parse numbered list: "1. ...", "2. ...", etc.
    sub_steps: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading "1.", "1)", "-", "*"
        cleaned = re.sub(r"^[\d]+[.)]\s*|^[-*]\s*", "", line).strip()
        if cleaned:
            sub_steps.append(cleaned)

    return sub_steps


def _fallback_decompose_instruction(instruction: str) -> list[str]:
    """Deterministic decomposition when LLM output is unavailable."""
    parts = [
        p.strip(" .")
        for p in re.split(r"\bthen\b|\band\b|,|;|->|=>", instruction, flags=re.IGNORECASE)
        if p.strip(" .")
    ]
    unique_parts: list[str] = []
    seen: set[str] = set()
    for p in parts:
        k = re.sub(r"\s+", " ", p).lower()
        if len(k) < 3 or k in seen:
            continue
        seen.add(k)
        unique_parts.append(p)
    if len(unique_parts) >= 2:
        return unique_parts[:5]
    # Cannot split further — return the instruction as a single atomic step.
    return [instruction.strip().rstrip(".")]


def _normalize_sequential_steps(instruction: str, llm_steps: list[str]) -> list[str]:
    """Return ordered, distinct sequential steps."""
    raw = llm_steps if llm_steps else _fallback_decompose_instruction(instruction)
    out: list[str] = []
    seen: set[str] = set()
    for step in raw:
        cleaned = re.sub(r"\s+", " ", step).strip(" .")
        key = cleaned.lower()
        if not cleaned or len(cleaned) < 3 or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= 5:
            break
    if not out:
        return _fallback_decompose_instruction(instruction)
    return out


class _ExplorationProgressEnv:
    """
    Thin env wrapper that renders a tqdm progress bar on sys.stderr while
    match_instruction() explores the environment.  Every step() call advances
    the bar by one; the postfix shows the count of unique states discovered.

    Written to stderr so it never touches the MCP stdout channel.
    """

    def __init__(self, env: Any, total_steps: int, env_id: str) -> None:
        import tqdm
        self._env = env
        self._unique_langs: set[str] = set()
        self._steps = 0
        self._bar = tqdm.tqdm(
            total=total_steps,
            desc=f"Exploring {env_id}",
            unit="step",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
        )

    def reset(self, seed: Any = None, options: Any = None) -> Any:
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        self._steps += 1
        self._bar.update(1)
        return result

    def close(self) -> None:
        self._bar.close()
        self._env.close()

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def env_id(self) -> str:
        return getattr(self._env, "env_id", "")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


@mcp.tool()
def rl_clear_obs_cache(env_id: str = "") -> str:
    """
    Clear the in-memory observation corpus cache used by rl_match_instruction.

    Call this when you have updated a language translator and want
    rl_match_instruction to rebuild the corpus with fresh translations,
    or when you want to force re-exploration of the state space.

    Parameters
    ----------
    env_id:
        Clear only the cache for this environment ID.
        When empty (default) the entire cache is cleared for all environments.

    Returns
    -------
    Confirmation of what was cleared.
    """
    from ..instruction_following import clear_obs_cache, obs_cache_info
    from ..language_translation.caching import clear_translation_cache, translation_cache_info

    before_obs = obs_cache_info()
    before_trans = translation_cache_info()

    target = env_id.strip() or None
    clear_obs_cache(target)
    clear_translation_cache(target)

    scope = f"'{target}'" if target else "all environments"
    obs_cleared = sum(before_obs[k] for k in (([target] if target else list(before_obs.keys()))) if k in before_obs)
    trans_cleared = sum(before_trans[k] for k in (([target] if target else list(before_trans.keys()))) if k in before_trans)

    return (
        f"Cache cleared for {scope}.\n"
        f"  Observation corpus entries removed: {obs_cleared}\n"
        f"  Translation cache entries removed:  {trans_cleared}\n\n"
        "Next call to rl_match_instruction will run fresh exploration."
    )


@mcp.tool()
async def rl_match_instruction(
    ctx: Context,
    env_id: str,
    instruction: str,
    exploration_steps: int = 100,
    seed: Optional[int] = None,
    top_k: int = 5,
    encoder: str = "tfidf",
    encoder_model: str = "",
    encoder_device: str = "",
) -> str:
    """
    Explore an RL environment, translate observed states to language, and
    find which observed state best matches a natural-language instruction.

    After exploration the tool calls the host LLM to decompose *instruction*
    into clear, distinct, ordered sub-steps grounded in the observed
    environment vocabulary. The first step is forced to focus on episode
    start behavior. These steps are then trained with the sequential
    instruction process.

    This is the first step of instruction-following RL.  After calling this
    tool you can run rl_instruction_run_episode() to train with the matched
    state(s) as sub-goals.

    Parameters
    ----------
    env_id:
        A registered RLIP environment ID, e.g. "Sailing-v0".
    instruction:
        The natural-language goal to match, e.g.
        "sail towards the beach side".
    exploration_steps:
        Number of random steps used to build the observation corpus.
        More steps give broader coverage; 50–200 is usually sufficient.
    seed:
        Optional integer seed for reproducible exploration.
    top_k:
        Number of top-ranked matches to include in the output (max 10).
    encoder:
        Text encoder for instruction-to-state similarity scoring.
        One of "tfidf" (default), "bm25", or "sentence-transformers".
        "sentence-transformers" uses a pre-trained neural model
        (all-MiniLM-L6-v2) for semantic similarity; requires the
        sentence-transformers package.
    encoder_model:
        Optional Hugging Face model id for sentence-transformers, e.g.
        "BAAI/bge-small-en-v1.5" or "sentence-transformers/all-mpnet-base-v2".
        Used only when encoder is sentence-transformers/sentence.
    encoder_device:
        Optional sentence-transformers device override ("cpu", "cuda").
        Used only when encoder is sentence-transformers/sentence.

    Returns
    -------
    A text summary of the best-matched state, its similarity score,
    the LLM-derived sequential sub-steps, and a match_id you can
    pass to rl_instruction_run_episode().
    """
    import uuid
    from ..instruction_following import (
        build_sequential_instruction_following_protocol,
        match_instruction,
    )
    from ..instruction_matching import get_encoder as _get_encoder
    from ..environments.registry import registry as _env_registry

    try:
        factory = _env_registry.get(env_id)
        env = factory.create()
    except KeyError:
        return (
            f"Environment '{env_id}' is not registered.  "
            "Call rl_list_environments() to see what is available."
        )

    progress_env = _ExplorationProgressEnv(env, total_steps=exploration_steps, env_id=env_id)

    encoder_model = encoder_model.strip()
    encoder_device = encoder_device.strip()

    # Persist the exact encoder spec so downstream training reuses the same
    # model when it rebuilds encoder instances.
    encoder_spec = encoder
    _ekey = encoder.lower().strip().replace("_", "-")
    if _ekey in {"sentence", "sentence-transformers", "hf"} and encoder_model:
        encoder_spec = f"sentence:{encoder_model}"

    async def _poll_exploration() -> None:
        while True:
            await asyncio.sleep(0.5)
            await ctx.report_progress(progress_env._steps, exploration_steps)

    poll_task = asyncio.create_task(_poll_exploration())
    try:
        try:
            _encoder_instance = _get_encoder(
                encoder_spec,
                sentence_model=(encoder_model or None),
                sentence_device=(encoder_device or None),
            )
            match = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: match_instruction(
                    instruction,
                    progress_env,
                    seed=seed,
                    max_steps=exploration_steps,
                    encoder=_encoder_instance,
                ),
            )
        except ValueError as exc:
            return f"Instruction matching failed: {exc}"
    finally:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass
        progress_env._bar.close()  # close bar; env itself stays open for protocol

    # ── LLM sub-goal decomposition ─────────────────────────────────────────────
    # Ask the host LLM to break the instruction into ordered sub-steps using the
    # observed language states as grounding context.  Falls back to [] silently
    # so the rest of the tool always runs even when sampling is unavailable.
    from ..instruction_following import obs_cache_langs
    observed_langs = obs_cache_langs(env_id)
    sub_steps_raw = await _decompose_instruction_with_llm(ctx, instruction, env_id, observed_langs)
    sub_steps = _normalize_sequential_steps(instruction, sub_steps_raw)

    # Each LLM-derived sub-step is an individual ordered instruction.  Build
    # the sequential protocol from sub_steps so that every sub-step is
    # independently matched to a language-translated observed state and used
    # in the ordered reward-shaping signal.
    protocol = build_sequential_instruction_following_protocol(
        sub_steps,
        env,
        seed=seed,
        max_steps=exploration_steps,
    )
    # The env was consumed by exploration; protocol will reset it on __call__.
    match_id = uuid.uuid4().hex[:12]
    _instruction_protocols[match_id] = {
        "protocol": protocol,
        "env_id":   env_id,
        "env":      env,
        "match":    match,
        "match_summary": {
            "instruction": instruction,
            "best_match_language": match.matched_language,
            "best_match_observation": match.matched_observation,
            "best_match_similarity": match.similarity_score,
        },
        "is_sequential": True,
        "instructions": sub_steps,
        "decomposition": sub_steps,
        "encoder_name": encoder_spec,
        # Original user-facing instruction (needed for plan DB lookups)
        "original_instruction": instruction,
    }

    # ── Instruction plan database ──────────────────────────────────────────────
    try:
        from ..instruction_plan_db import get_plan_database
        from ._state import _env_plan_db_path
        _plan_db = get_plan_database(env_id, plan_path=str(_env_plan_db_path(env_id)))
        _plan_db.record_instruction_use(
            instruction=instruction,
            match_id=match_id,
            source="llm",
            sub_steps=sub_steps,
            similarity=match.similarity_score,
            matched_language=match.matched_language,
        )
    except Exception:
        pass

    top_k = max(1, min(top_k, 10))
    top_lines = [
        f"  {i+1}. sim={sc:.4f}  {lg[:100]}"
        for i, (lg, sc) in enumerate(match.all_scores[:top_k])
    ]

    steps_block = "Decomposed sequential steps:\n" + "\n".join(
        f"  {i+1}. {s}" for i, s in enumerate(sub_steps)
    ) + "\n\n"

    return (
        f"Instruction matched for '{env_id}':\n\n"
        f"  Instruction:   {instruction!r}\n"
        f"  Encoder:       {encoder_spec}\n"
        f"  Best match:    {match.matched_language}\n"
        f"  Similarity:    {match.similarity_score:.4f}\n"
        f"  Match ID:      {match_id}\n\n"
        + steps_block
        + f"Top {top_k} candidates:\n" + "\n".join(top_lines) + "\n\n"
        f"Use rl_instruction_run_episode(match_id='{match_id}') to run a "
        f"training episode with sequential instruction shaping."
    )


@mcp.tool()
def rl_instruction_run_episode(
    match_id: str,
    max_steps: int = 200,
    sub_goal_bonus: float = 0.0,
    sub_goal_threshold: float = 0.5,
    sub_goal_repeatable: bool = False,
    seed: Optional[int] = None,
) -> str:
    """
    Run a sub-goal-shaped RL training episode using a matched instruction.

    Must be called after rl_match_instruction().  The previously matched
    state acts as a language-grounded sub-goal: whenever the agent's
    observed state has a cosine similarity ≥ sub_goal_threshold to the
    sub-goal description, an additional bonus reward is added.

    Parameters
    ----------
    match_id:
        The match_id returned by rl_match_instruction().
    max_steps:
        Maximum steps for the training episode.
    sub_goal_bonus:
        Bonus reward added when the language similarity threshold is met.
        Set to 0.0 (default) to auto-scale: ``max_reward / (100 × n_sub_goals)``
        where *max_reward* is inferred from the environment's reward range and
        *n_sub_goals* is the number of matched states.  Pass an explicit
        positive value to override.
    sub_goal_threshold:
        Cosine similarity threshold (0–1) required to award the bonus.
        Lower values make the sub-goal easier to reach.
    sub_goal_repeatable:
        False (default) – bonus awarded at most once per episode.
        True – bonus awarded on every step the threshold is met.
    seed:
        Optional seed for the training episode reset.

    Returns
    -------
    A detailed summary of the episode including total reward,
    which step(s) the sub-goal was reached, and a step-by-step
    language trajectory excerpt (up to 10 steps shown).
    """
    entry = _instruction_protocols.get(match_id)
    if entry is None:
        return (
            f"Match ID '{match_id}' not found.  "
            "Run rl_match_instruction() first to obtain a valid match_id."
        )

    protocol = entry["protocol"]
    env = entry["env"]

    # Route sequential entries to the sequential execution path.
    if entry.get("is_sequential"):
        return rl_sequential_instruction_run_episode(
            match_id=match_id,
            max_steps=max_steps,
            sub_goal_bonus=sub_goal_bonus,
            sub_goal_threshold=sub_goal_threshold,
            seed=seed,
        )

    # Auto-scale bonus when caller passed 0.0 (the sentinel for "auto").
    if sub_goal_bonus == 0.0:
        from ..instruction_following import scale_sub_goal_bonus  # noqa: PLC0415
        n_sub_goals = len(protocol._all_sub_goal_languages)
        sub_goal_bonus = scale_sub_goal_bonus(env, n_instructions=n_sub_goals)

    # Apply per-call overrides
    protocol.max_steps = max_steps
    protocol.sub_goal_bonus = sub_goal_bonus
    protocol.sub_goal_threshold = sub_goal_threshold
    protocol.sub_goal_repeatable = sub_goal_repeatable
    if seed is not None:
        protocol.seed = seed

    result = protocol(env)
    ep = result.episodes[0]

    sub_goal_steps = [
        r.step for r in ep.history if r.info.get("sub_goal_reached")
    ]
    sim_values = [
        r.info.get("sub_goal_similarity")
        for r in ep.history
        if r.info.get("sub_goal_similarity") is not None
    ]
    max_sim = max(sim_values) if sim_values else None

    # Build a readable trajectory excerpt (first 10 steps)
    excerpt_lines: list[str] = []
    for rec in ep.history[:10]:
        sim = rec.info.get("sub_goal_similarity", "n/a")
        hit = " ◀ sub-goal" if rec.info.get("sub_goal_reached") else ""
        lang = (rec.language_obs or "")[:80]
        excerpt_lines.append(
            f"  step {rec.step:3d}  r={rec.reward:+.3f}  sim={sim}  {lang}{hit}"
        )
    if len(ep.history) > 10:
        excerpt_lines.append(f"  … ({len(ep.history) - 10} more steps not shown)")

    peak_line = f"\n  Peak similarity: {max_sim:.4f}" if max_sim is not None else ""

    return (
        f"Instruction-following episode complete\n"
        f"  Environment:   {result.env_id}\n"
        f"  Instruction:   {protocol.instruction!r}\n"
        f"  Sub-goal:      {protocol.sub_goal_language[:80]}\n"
        f"  Threshold:     {sub_goal_threshold}  Bonus: {sub_goal_bonus:.6g}\n\n"
        f"  Steps:         {ep.steps}\n"
        f"  Total reward:  {ep.total_reward:.4f}\n"
        f"  End reason:    {ep.end_reason}\n"
        f"  Sub-goal reached at steps: "
        + (", ".join(str(s) for s in sub_goal_steps) if sub_goal_steps else "never")
        + peak_line
        + "\n\nTrajectory excerpt:\n" + "\n".join(excerpt_lines)
    )


@mcp.tool()
async def rl_match_sequential_instructions(
    ctx: Context,
    env_id: str,
    instructions: list[str],
    exploration_steps: int = 100,
    seed: Optional[int] = None,
) -> str:
    """
    Explore an RL environment and match multiple natural-language instructions
    that must be completed in sequence.

    This tool automatically:
    1. Explores the environment and builds a language state corpus
    2. Matches each instruction to observed states via TF-IDF similarity
    3. Uses the host LLM to decompose each instruction into clear sub-steps
    4. Sets up sequential reward shaping for training

    The environment is explored once and all instructions are matched against
    the same observation corpus, making this efficient for decomposed tasks.
    Each instruction is automatically decomposed into sub-steps grounded in
    the environment's actual vocabulary.

    After calling this tool, use rl_sequential_instruction_run_episode() with
    the returned match_id to train with sequential sub-goal shaping.

    Parameters
    ----------
    env_id:
        A registered RLIP environment ID, e.g. "Sailing-v0".
    instructions:
        List of natural-language instructions in completion order, e.g.
        ["sail towards the beach", "approach the dock", "return to harbor"].
    exploration_steps:
        Number of random steps used to build the observation corpus.
        50–200 is usually sufficient.
    seed:
        Optional integer seed for reproducible exploration.

    Returns
    -------
    A text summary of each instruction's decomposition, best match, and
    similarity scores. Includes a match_id for use with
    rl_sequential_instruction_run_episode().
    """
    import uuid
    from ..instruction_following import (
        build_sequential_instruction_following_protocol,
        match_instruction,
        obs_cache_langs,
    )
    from ..environments.registry import registry as _env_registry

    if not instructions:
        return "Error: instructions list cannot be empty."

    try:
        factory = _env_registry.get(env_id)
        env = factory.create()
    except KeyError:
        return (
            f"Environment '{env_id}' is not registered.  "
            "Call rl_list_environments() to see what is available."
        )

    progress_env = _ExplorationProgressEnv(env, total_steps=exploration_steps, env_id=env_id)

    async def _poll_exploration() -> None:
        while True:
            await asyncio.sleep(0.5)
            await ctx.report_progress(progress_env._steps, exploration_steps)

    poll_task = asyncio.create_task(_poll_exploration())
    try:
        try:
            # Match all instructions (first one runs exploration, rest reuse cache)
            matches: list[Any] = []
            for idx, instr in enumerate(instructions):
                match = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda i=instr, first=(idx == 0): match_instruction(
                        i,
                        progress_env if first else env,
                        seed=seed,
                        max_steps=exploration_steps,
                    ),
                )
                matches.append(match)
        except ValueError as exc:
            return f"Instruction matching failed: {exc}"
    finally:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass
        progress_env._bar.close()

    # The caller-provided instructions are already the ordered sequential steps —
    # no further LLM decomposition is needed.  Deduplicate while preserving order.
    sequential_instructions: list[str] = []
    seen_seq: set[str] = set()
    for instr in instructions:
        key = instr.strip().lower()
        if key and key not in seen_seq:
            seen_seq.add(key)
            sequential_instructions.append(instr.strip())

    protocol = build_sequential_instruction_following_protocol(
        sequential_instructions,
        env,
        seed=seed,
        max_steps=exploration_steps,
    )

    match_id = uuid.uuid4().hex[:12]
    _instruction_protocols[match_id] = {
        "protocol": protocol,
        "env_id": env_id,
        "env": env,
        "matches": matches,
        "is_sequential": True,
        "instructions": sequential_instructions,
        "original_instructions": instructions,
    }

    # Build output showing each instruction and its best match
    output_lines = [
        f"Sequential instructions matched for '{env_id}':\n"
    ]

    for idx, (instr, match) in enumerate(zip(sequential_instructions, matches)):
        output_lines.append(
            f"\n  {idx + 1}. Instruction: {instr!r}\n"
            f"     Best match:  {match.matched_language}\n"
            f"     Similarity:  {match.similarity_score:.4f}\n"
            f"     Sub-goals:   {len(match.matched_states)} state(s)"
        )
        output_lines.append("")

    output_lines.append(
        f"\n  Match ID: {match_id}\n\n"
        f"  Effective sequential steps: {len(sequential_instructions)}\n"
        f"Use rl_sequential_instruction_run_episode(match_id='{match_id}') to run a\n"
        f"training episode that completes these steps in sequence."
    )

    return "\n".join(output_lines)


@mcp.tool()
def rl_sequential_instruction_run_episode(
    match_id: str,
    max_steps: int = 200,
    sub_goal_bonus: float = 0.0,
    sub_goal_threshold: float = 0.5,
    seed: Optional[int] = None,
) -> str:
    """
    Run a sequential multi-step instruction-following RL training episode.

    Must be called after rl_match_sequential_instructions().  The protocol
    applies reward shaping for each instruction in sequence:

    1. Initially, only the first instruction's sub-goal receives reward
    2. When the agent reaches the first instruction's sub-goal, it advances to
       the second instruction
    3. Shaping reward now applies only to the second instruction's sub-goal
    4. Process continues until all instructions are completed or max_steps reached

    This enables learning of multi-step behaviors with progressive sub-goal
    shaping at each stage.

    Parameters
    ----------
    match_id:
        The match_id returned by rl_match_sequential_instructions().
    max_steps:
        Maximum steps for the training episode.
    sub_goal_bonus:
        Bonus reward per instruction when sub-goal is reached.
        Set to 0.0 (default) to auto-scale: ``max_reward / (100 × n_instructions)``
        where *max_reward* is inferred from the environment's reward range.
        Pass an explicit positive value to override.
    sub_goal_threshold:
        Cosine similarity threshold (0–1) required to award the bonus.
        Lower values make sub-goals easier to reach.
    seed:
        Optional seed for the training episode reset.

    Returns
    -------
    A detailed summary of the episode including which instructions were
    completed, step counts, total reward, and a trajectory excerpt showing
    when each instruction was reached.
    """
    entry = _instruction_protocols.get(match_id)
    if entry is None or not entry.get("is_sequential"):
        return (
            f"Match ID '{match_id}' not found or is not sequential.  "
            "Run rl_match_sequential_instructions() first to obtain a valid match_id."
        )

    protocol = entry["protocol"]
    env = entry["env"]
    instructions = entry.get("instructions", [])

    # Auto-scale bonus when caller passed 0.0 (the sentinel for "auto").
    if sub_goal_bonus == 0.0:
        from ..instruction_following import scale_sub_goal_bonus  # noqa: PLC0415
        sub_goal_bonus = scale_sub_goal_bonus(env, n_instructions=len(instructions))

    # Apply per-call overrides
    protocol.max_steps = max_steps
    protocol.sub_goal_bonus = sub_goal_bonus
    protocol.sub_goal_threshold = sub_goal_threshold
    if seed is not None:
        protocol.seed = seed

    result = protocol(env)
    ep = result.episodes[0]

    # Collect information about instruction completion
    instr_reached_steps: dict[int, int] = {}  # instruction_idx → step_reached
    for rec in ep.history:
        if rec.info.get("sub_goal_reached"):
            instr_idx = rec.info.get("current_instruction", 0)
            if instr_idx not in instr_reached_steps:
                instr_reached_steps[instr_idx] = rec.step

    n_complete = result.metadata.get("instructions_completed", 0)

    # Build a readable trajectory excerpt showing instruction progression
    excerpt_lines: list[str] = []
    for rec in ep.history[:15]:
        curr_instr = rec.info.get("current_instruction", 0)
        sim = rec.info.get("sub_goal_similarity", "n/a")
        hit = " ◀ instruction reached" if rec.info.get("sub_goal_reached") else ""
        lang = (rec.language_obs or "")[:70]
        instr_label = f"[instr {curr_instr + 1}]" if curr_instr < len(instructions) else "[complete]"
        excerpt_lines.append(
            f"  step {rec.step:3d}  r={rec.reward:+.3f}  sim={sim}  {instr_label}  {lang}{hit}"
        )
    if len(ep.history) > 15:
        excerpt_lines.append(f"  … ({len(ep.history) - 15} more steps not shown)")

    # Format instruction completion summary
    completion_lines = []
    for idx, instr in enumerate(instructions):
        if idx in instr_reached_steps:
            step = instr_reached_steps[idx]
            completion_lines.append(f"    {idx + 1}. ✓ {instr[:70]}  (reached at step {step})")
        else:
            completion_lines.append(f"    {idx + 1}. ✗ {instr[:70]}  (not reached)")

    return (
        f"Sequential instruction-following episode complete\n"
        f"  Environment:       {result.env_id}\n"
        f"  Total instructions: {len(instructions)}\n"
        f"  Completed:         {n_complete} / {len(instructions)}\n"
        f"  Threshold:         {sub_goal_threshold}  Bonus: {sub_goal_bonus:.6g}\n\n"
        f"  Steps:             {ep.steps}\n"
        f"  Total reward:      {ep.total_reward:.4f}\n"
        f"  End reason:        {ep.metadata.get('end_reason', 'unknown')}\n\n"
        f"Instructions:\n" + "\n".join(completion_lines) +
        f"\n\nTrajectory excerpt:\n" + "\n".join(excerpt_lines)
    )


@mcp.tool()
def rl_get_instruction_plan(env_id: str) -> str:
    """
    Show the instruction planning database for an environment.

    Returns a summary of all instructions that have been tried in this
    environment, including:
    - Source: whether each instruction was provided by the user/LLM ('llm'),
      or automatically derived from a training run ('derived').
    - Times used: how many training runs used this instruction.
    - BestEval: the best clean evaluation reward seen after any training run
      that used this instruction (measured WITHOUT any instruction-shaping
      bonus, so it is an unbiased performance measure).
    - DrvScore: for derived instructions, the CSR × log₂(1+visits) score
      from rl_train_and_derive_instructions().
    - Similarity: cosine similarity between the instruction text and its
      best-matching observed environment state.

    Use this tool to:
    - Plan which instructions to try next based on past evaluation rewards.
    - Understand which instructions have already been attempted.
    - Compare the effectiveness of user-specified vs. derived instructions.
    - Advise the user on whether RL training is making progress.

    Parameters
    ----------
    env_id:
        A registered RLIP environment ID, e.g. "Sailing-v0".

    Returns
    -------
    A formatted table of all known instructions and their outcomes.
    """
    from ..instruction_plan_db import get_plan_database, _PLAN_DATABASES
    from ._state import _env_plan_db_path

    plan_path = str(_env_plan_db_path(env_id))

    # Force a fresh load if there is no in-memory instance yet (MCP server
    # restart between sessions, or first call after a fresh training run).
    if env_id not in _PLAN_DATABASES:
        db = get_plan_database(env_id, plan_path=plan_path)
    else:
        db = _PLAN_DATABASES[env_id]
        # Re-load from disk in case another tool (e.g. rl_train_agent) wrote
        # new eval_reward data outside this process.
        db._load()

    if not db._entries:
        return (
            f"No instruction plan data found for '{env_id}'.\n\n"
            "Run rl_match_instruction() or rl_train_and_derive_instructions() "
            "first to build the database."
        )

    return db.summary_text()


