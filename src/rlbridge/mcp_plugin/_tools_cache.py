"""
Environment cache MCP tools for the rlbridge plugin.

Tools: rl_load_cached_environments, rl_list_cached_environments,
       rl_clear_obs_cache.
"""

from __future__ import annotations

import json

from ._state import (
    _custom_env_cache_root,
    _custom_translators,
    mcp,
)


@mcp.tool()
def rl_load_cached_environments() -> str:
    """
    Load all custom environments previously built with rl_build_environment()
    from ``~/.rlbridge/environments/`` and register them into this session.

    Call this once at the start of a session to restore environments that were
    created in a previous session.  Already-registered environments are
    re-registered without error (the latest cached version takes precedence).

    Returns
    -------
    A list of loaded environment IDs and their translator status.
    """
    from ..environments.builder import load_cached_environments  # noqa: PLC0415

    try:
        loaded = load_cached_environments(cache_dir=_custom_env_cache_root())
    except Exception as exc:
        return f"Failed to load cached environments: {exc}"

    if not loaded:
        return (
            f"No cached environments found in {_custom_env_cache_root()}.\n"
            "Use rl_build_environment() to create and cache a new environment."
        )

    # Also refresh in-process translator cache
    for built in loaded:
        if built.translator is not None:
            _custom_translators[built.spec.env_id] = built.translator

    lines = [f"Loaded {len(loaded)} cached environment(s):\n"]
    for built in loaded:
        has_t = "yes" if built.translator else "no"
        lines.append(
            f"  • {built.spec.env_id:40s}  translator={has_t}  "
            f"namespace={built.spec.namespace}"
        )
    return "\n".join(lines)


@mcp.tool()
def rl_list_cached_environments() -> str:
    """
    List all custom environments stored in the rlbridge cache (``~/.rlbridge/environments/``).

    Does not register them - call rl_load_cached_environments() to register.

    Returns summary metadata for each cached environment.
    """
    root = _custom_env_cache_root()
    if not root.exists():
        return (
            f"No cached environments found ({root} does not exist).\n"
            "Use rl_build_environment() to create your first custom environment."
        )

    entries: list[dict] = []
    for env_dir in sorted(root.iterdir()):
        spec_path = env_dir / "spec.json"
        if not spec_path.exists():
            continue
        try:
            entries.append(json.loads(spec_path.read_text()))
        except Exception:
            continue

    if not entries:
        return "No cached environments found."

    lines = [f"Cached environments ({len(entries)}):\n"]
    for spec in entries:
        has_t = "yes" if spec.get("language_translation") else "no"
        lines.append(
            f"  • {spec['env_id']}\n"
            f"    {spec.get('description', '(no description)')}\n"
            f"    tags={spec.get('tags', [])}  namespace={spec.get('namespace', '')}  "
            f"translator={has_t}\n"
            f"    created: {spec.get('created_at', '?')[:19]}"
        )
    return "\n".join(lines)


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
