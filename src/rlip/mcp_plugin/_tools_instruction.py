"""
Instruction-following MCP tools for the RLIP plugin.

Tools: rl_clear_obs_cache, rl_match_instruction, rl_instruction_run_episode.

Uses the shared ``_instruction_protocols`` cache from ``_state`` so that
rl_train_agent (in _tools_agents) can access previously-matched sub-goals.
"""

from __future__ import annotations

from typing import Any, Optional

from ._state import _instruction_protocols, mcp


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
def rl_match_instruction(
    env_id: str,
    instruction: str,
    exploration_steps: int = 100,
    seed: Optional[int] = None,
    top_k: int = 5,
) -> str:
    """
    Explore an RL environment, translate observed states to language, and
    find which observed state best matches a natural-language instruction
    using TF-IDF cosine similarity.

    This is the first step of instruction-following RL.  After calling this
    tool you can run rl_instruction_run_episode() to train with the matched
    state as a sub-goal.

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

    Returns
    -------
    A text summary of the best-matched state, its similarity score, and
    a match_id you can pass to rl_instruction_run_episode().
    """
    import uuid
    from ..instruction_following import match_instruction, build_instruction_following_protocol
    from ..environments.registry import registry as _env_registry

    try:
        factory = _env_registry.get(env_id)
        env = factory.create()
    except KeyError:
        return (
            f"Environment '{env_id}' is not registered.  "
            "Call rl_list_environments() to see what is available."
        )

    try:
        match = match_instruction(
            instruction,
            env,
            seed=seed,
            max_steps=exploration_steps,
        )
    except ValueError as exc:
        return f"Instruction matching failed: {exc}"

    # Build and cache the ready-to-run protocol so the agent can immediately
    # call rl_instruction_run_episode without re-running exploration.
    protocol = build_instruction_following_protocol(
        instruction,
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
    }

    top_k = max(1, min(top_k, 10))
    top_lines = [
        f"  {i+1}. sim={sc:.4f}  {lg[:100]}"
        for i, (lg, sc) in enumerate(match.all_scores[:top_k])
    ]

    return (
        f"Instruction matched for '{env_id}':\n\n"
        f"  Instruction:   {instruction!r}\n"
        f"  Best match:    {match.matched_language}\n"
        f"  Similarity:    {match.similarity_score:.4f}\n"
        f"  Match ID:      {match_id}\n\n"
        f"Top {top_k} candidates:\n" + "\n".join(top_lines) + "\n\n"
        f"Use rl_instruction_run_episode(match_id='{match_id}') to run a "
        f"training episode with this state as a sub-goal."
    )


@mcp.tool()
def rl_instruction_run_episode(
    match_id: str,
    max_steps: int = 200,
    sub_goal_bonus: float = 1.0,
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
        f"  Threshold:     {sub_goal_threshold}\n\n"
        f"  Steps:         {ep.steps}\n"
        f"  Total reward:  {ep.total_reward:.4f}\n"
        f"  End reason:    {ep.end_reason}\n"
        f"  Sub-goal reached at steps: "
        + (", ".join(str(s) for s in sub_goal_steps) if sub_goal_steps else "never")
        + peak_line
        + "\n\nTrajectory excerpt:\n" + "\n".join(excerpt_lines)
    )
