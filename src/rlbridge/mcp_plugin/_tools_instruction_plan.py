"""
Instruction plan and cache MCP tools for the RLIP plugin.

Tools: rl_get_instruction_plan, rl_list_cached_instructions,
       rl_clear_instruction_cache.
"""

from __future__ import annotations

from ._state import _env_plan_db_path, mcp


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


@mcp.tool()
def rl_list_cached_instructions(env_id: str = "") -> str:
    """
    List all cached instruction entries for one environment (or all
    environments), showing each instruction's matched state, similarity
    score, and accumulated episode success statistics.

    Instructions are populated by:
      • rl_match_instruction() - one entry per explicit user instruction
      • rl_train_and_derive_instructions() - auto-derived entries

    Parameters
    ----------
    env_id:
        Restrict output to this environment.  When empty, all environments
        are shown.

    Returns
    -------
    A text table of cached instructions with success-rate statistics.
    """
    from ..instruction_following import list_instruction_cache

    cache = list_instruction_cache(env_id.strip() or None)
    if not cache:
        scope = f"'{env_id}'" if env_id.strip() else "any environment"
        return f"No cached instructions found for {scope}."

    lines: list[str] = []
    for eid, entries in sorted(cache.items()):
        lines.append(f"Environment: {eid}  ({len(entries)} instruction(s))\n")
        for instr, entry in entries.items():
            sr = f"{entry.success_rate:.1%}" if entry.episodes_run > 0 else "n/a"
            lines.append(
                f"  instruction:   {instr!r}\n"
                f"  matched state: {entry.match.matched_language!r}\n"
                f"  similarity:    {entry.match.similarity_score:.4f}\n"
                f"  episodes_run:  {entry.episodes_run}  "
                f"successes={entry.successes}  success_rate={sr}\n"
            )
        lines.append("")

    lines.append(
        "Use rl_apply_derived_instruction(env_id, instruction) to turn any of "
        "these into a match_id for rl_instruction_run_episode() or rl_train_agent()."
    )
    return "\n".join(lines)


@mcp.tool()
def rl_clear_instruction_cache(env_id: str = "") -> str:
    """
    Clear the instruction cache (and the state→instruction map) for one
    environment or all environments.

    Parameters
    ----------
    env_id:
        Clear only this environment.  When empty the entire cache is cleared.

    Returns
    -------
    Confirmation string.
    """
    from ..instruction_following import (
        clear_instruction_cache,
        list_instruction_cache,
    )

    target = env_id.strip() or None
    before = list_instruction_cache(target)
    total = sum(len(v) for v in before.values())
    clear_instruction_cache(target)

    scope = f"'{target}'" if target else "all environments"
    return (
        f"Instruction cache cleared for {scope}.\n"
        f"Removed {total} cached instruction(s)."
    )
