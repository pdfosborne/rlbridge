"""
Environment builder MCP tools for the RLIP plugin.

Tools: rl_build_environment, rl_load_cached_environments,
       rl_list_cached_environments, rl_sample_states_for_translation,
       rl_set_translator_code, rl_translate_state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ._state import _custom_translators, _sampled_states, mcp


@mcp.tool()
def rl_build_environment(
    env_id: str,
    gym_env_id: str,
    description: str = "",
    tags: str = "",
    namespace: str = "custom",
    max_episode_steps: int = 0,
    translator_name: str = "",
) -> str:
    """
    Wrap a Gymnasium environment with custom metadata, cache it locally,
    and register it so it is immediately available in this session.

    The environment is saved to ``~/.rlip/envs/<env_id>/`` and written to
    ``~/.rlip/catalog.json`` so it is reloaded automatically on restart via
    rl_load_cached_environments().

    Parameters
    ----------
    env_id:
        The identifier to register this environment under (e.g.
        "FrozenLake-Custom-v0").  Can differ from gym_env_id.
    gym_env_id:
        The Gymnasium environment ID to wrap (e.g. "FrozenLake-v1").
    description:
        Human-readable description shown in rl_list_environments().
    tags:
        Comma-separated tags (e.g. "grid,discrete,custom").
    namespace:
        Registry namespace (default "custom").
    max_episode_steps:
        Hard episode step limit.  0 = use the Gymnasium default.
    translator_name:
        Optional name of a registered translator to attach (e.g.
        "Sailing-v0").  Leave blank to add a translator later with
        rl_set_translator_code().

    Returns
    -------
    Confirmation plus the local cache path.
    """
    from ..environments.builder import EnvironmentBuilder  # noqa: PLC0415

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    builder = (
        EnvironmentBuilder(env_id)
        .from_gymnasium(gym_env_id)
        .with_metadata(
            description=description,
            tags=tag_list,
            namespace=namespace,
            max_episode_steps=max_episode_steps or None,
        )
    )

    if translator_name:
        try:
            builder = builder.with_translator(translator_name)
        except ValueError as exc:
            return (
                f"Translator '{translator_name}' not found: {exc}\n"
                "Building environment without a translator.  You can add one "
                "later with rl_set_translator_code()."
            )

    try:
        built = builder.build()
    except Exception as exc:
        return f"Failed to build environment '{env_id}': {exc}"

    translator_line = (
        f"  Translator: {type(built.translator).__name__}"
        if built.translator
        else "  Translator: none  (use rl_set_translator_code() to add one)"
    )

    return (
        f"Environment '{env_id}' built and registered.\n\n"
        f"  Wraps:            {gym_env_id}\n"
        f"  Description:      {description or '(none)'}\n"
        f"  Tags:             {', '.join(tag_list) or '(none)'}\n"
        f"  Namespace:        {namespace}\n"
        f"{translator_line}\n"
        f"  Cache path:       {built.cache_path}\n\n"
        f"Use rl_create(env_id='{env_id}') to create an instance, or\n"
        f"rl_sample_states_for_translation(env_id='{env_id}') to start "
        f"building a language translator."
    )


@mcp.tool()
def rl_load_cached_environments() -> str:
    """
    Load all custom environments previously built with rl_build_environment()
    from ``~/.rlip/envs/`` and register them into this session.

    Call this once at the start of a session to restore environments that were
    created in a previous session.  Already-registered environments are
    re-registered without error (the latest cached version takes precedence).

    Returns
    -------
    A list of loaded environment IDs and their translator status.
    """
    from ..environments.builder import load_cached_environments  # noqa: PLC0415

    try:
        loaded = load_cached_environments()
    except Exception as exc:
        return f"Failed to load cached environments: {exc}"

    if not loaded:
        return (
            "No cached environments found in ~/.rlip/envs/.\n"
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
    List all custom environments stored in the local cache (``~/.rlip/envs/``).

    Does not register them — call rl_load_cached_environments() to register.

    Returns summary metadata for each cached environment.
    """
    from ..environments.builder import _DEFAULT_CACHE_DIR  # noqa: PLC0415

    root = _DEFAULT_CACHE_DIR
    if not root.exists():
        return (
            "No cached environments found (~/.rlip/envs/ does not exist).\n"
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
def rl_sample_states_for_translation(
    env_id: str,
    n_samples: int = 20,
    seed: Optional[int] = None,
) -> str:
    """
    Sample unique observed states from an environment to help you write a
    language translator.

    Runs random exploration and collects up to *n_samples* distinct
    observations.  The sampled states are held in memory so you can pass
    ``rl_set_translator_code()`` immediately afterward without repeating them.

    Workflow
    --------
    1. Call this tool to see what the raw state values look like.
    2. Write a Python ``translate(state, ...)`` function that maps each state
       format to a natural-language description.
    3. Call ``rl_set_translator_code(env_id=..., python_code=...)`` to install
       and test it.

    Parameters
    ----------
    env_id:
        A registered RLIP environment ID.
    n_samples:
        Number of unique states to collect (default 20, max 50).
    seed:
        Optional seed for reproducible sampling.
    """
    from ..environments.registry import registry as _env_registry  # noqa: PLC0415
    from ..language_translation.generator import _sample_states  # noqa: PLC0415

    n_samples = min(n_samples, 50)

    try:
        factory = _env_registry.get(env_id)
        env = factory.create()
    except Exception as exc:
        return f"Could not create environment '{env_id}': {exc}"

    try:
        samples = _sample_states(
            env,
            n_samples=n_samples,
            max_steps_per_episode=50,
            max_episodes=n_samples * 10,
            seed=seed,
        )
    except Exception as exc:
        return f"State sampling failed: {exc}"
    finally:
        try:
            env.close()
        except Exception:
            pass

    _sampled_states[env_id] = samples

    collected = len(samples)
    header = (
        f"Sampled {collected} unique states from '{env_id}'."
        if collected >= n_samples
        else f"Sampled {collected}/{n_samples} unique states from '{env_id}' "
             f"(episode/step budget exhausted — this is normal for large or "
             f"complex environments)."
    )
    lines = [
        f"{header}\n",
        "Write a translate() function for these states, then call\n"
        f"rl_set_translator_code(env_id='{env_id}', python_code='...').\n",
        "Sampled states:",
    ]
    for i, (obs, ah) in enumerate(samples, 1):
        last = f"  (last action: {ah[-1]})" if ah else ""
        lines.append(f"  STATE {i:2d}: {repr(obs)}{last}")

    lines += [
        "",
        f"Example translate() function skeleton for '{env_id}':",
        "  def translate(state, *, legal_moves=None, action_history=None):",
        "      # Parse state and return a description string",
        "      # Return \"\" for states you cannot handle",
        f"      return f\"State: {{state}}\"",
    ]
    return "\n".join(lines)


@mcp.tool()
def rl_set_translator_code(
    env_id: str,
    python_code: str,
    save: bool = True,
) -> str:
    """
    Install a Python language translator function for an environment.

    Write a ``translate(state, *, legal_moves=None, action_history=None)``
    function and pass it here.  The translator is compiled and registered
    immediately so all subsequent protocol calls that use language translation
    (``translate=True``) pick it up automatically.

    Parameters
    ----------
    env_id:
        Environment to attach the translator to (must be registered with RLIP).
    python_code:
        Complete Python source of a ``translate`` function.  Example::

            def translate(state, *, legal_moves=None, action_history=None):
                parts = str(state).split("_")
                if len(parts) == 2:
                    return f"Position {parts[0]}, angle {parts[1]}"
                return ""

        Rules:
        - Must define a function named ``translate``.
        - Return a non-empty string for known states.
        - Return ``""`` (or raise) for states the function cannot handle —
          the system will fall back gracefully.
        - Do not import external packages; only Python built-ins are safe.
    save:
        If True (default), persist the translator to
        ``~/.rlip/envs/<env_id>/translator.py`` and update ``spec.json``
        so it is reloaded automatically by ``rl_load_cached_environments()``.

    Returns
    -------
    Confirmation plus test translations against previously sampled states
    (if rl_sample_states_for_translation was called for this env_id).
    """
    import datetime as _dt  # noqa: PLC0415

    from ..language_translation.generator import (  # noqa: PLC0415
        GeneratedTranslator,
        _strip_markdown_fences,
        _to_class_name,
    )
    from ..language_translation import TRANSLATORS  # noqa: PLC0415

    code = _strip_markdown_fences(python_code)

    def _no_llm(prompt: str) -> str:
        return ""

    try:
        gt = GeneratedTranslator(
            llm_fn=_no_llm,
            env_id=env_id,
            rule_code=code,
            env_context="",
            refine_threshold=99_999,  # disable auto-refine for hand-written rules
        )
    except ValueError as exc:
        return (
            f"Could not compile translator code for '{env_id}': {exc}\n\n"
            "Ensure the code defines a function with this exact signature:\n"
            "  def translate(state, *, legal_moves=None, action_history=None): ..."
        )

    _custom_translators[env_id] = gt
    TRANSLATORS[env_id] = gt

    cache_msg = ""
    if save:
        cache_dir = Path.home() / ".rlip" / "envs" / env_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        class_name = _to_class_name(env_id)
        module_path = cache_dir / "translator.py"
        gt.save_code(module_path, class_name=class_name)

        spec_path = cache_dir / "spec.json"
        if spec_path.exists():
            try:
                spec = json.loads(spec_path.read_text())
                spec["language_translation"] = {
                    "type": "generated",
                    "module_path": "translator.py",
                    "class_name": class_name,
                }
                spec["updated_at"] = _dt.datetime.now().isoformat()
                spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            except Exception as exc:
                cache_msg = f"\n(spec.json update failed: {exc})"
        cache_msg = f"\nSaved to: {module_path}" + cache_msg

    # Test against previously sampled states (up to 8)
    test_lines: list[str] = []
    if env_id in _sampled_states:
        test_lines.append("\nTest translations on sampled states:")
        for obs, _ah in _sampled_states[env_id][:8]:
            try:
                desc = gt.translate(obs)
                result_str = repr(desc) if desc else '""  ← fallback will apply'
            except Exception as exc:
                result_str = f"ERROR: {exc}"
            test_lines.append(f"  {str(repr(obs)):42s} → {result_str}")

    return (
        f"Translator installed for '{env_id}'.\n"
        f"  Compiled rule function: yes\n"
        f"  Auto-registered in TRANSLATORS: yes"
        + cache_msg
        + ("\n" + "\n".join(test_lines) if test_lines else "")
        + f"\n\nUse translate=True in any protocol call, or test with:\n"
        f"  rl_translate_state(env_id='{env_id}', state='...')"
    )


@mcp.tool()
def rl_translate_state(
    env_id: str,
    state: str,
) -> str:
    """
    Translate a raw environment observation to its natural-language description.

    Useful for inspecting translation quality or debugging a translator before
    using it in instruction-matching or protocol calls.

    Parameters
    ----------
    env_id:
        A registered RLIP environment ID.
    state:
        The raw state value encoded as a JSON string.
        - String observations: ``'"0.0300_0.2"'``
        - Integer observations: ``"3"``
        - List/array observations: ``"[1, 0, 2]"``

    Returns
    -------
    The natural-language description produced by the registered translator.
    """
    from ..language_translation import get_translator  # noqa: PLC0415

    translator = _custom_translators.get(env_id) or get_translator(env_id)
    if translator is None:
        return (
            f"No translator registered for '{env_id}'.\n"
            "Use rl_set_translator_code() to install one, or\n"
            "rl_sample_states_for_translation() to start the workflow."
        )

    try:
        obs = json.loads(state)
    except json.JSONDecodeError:
        obs = state  # treat as a raw string

    try:
        description = translator.translate(obs)
    except Exception as exc:
        return f"Translation raised an exception: {exc}"

    if not description:
        return (
            f"Translator returned empty string for state {state!r}.\n"
            "The state may be out of range or in an unrecognised format.\n"
            "Update rl_set_translator_code() to handle this state."
        )

    return f"Translation for {env_id} | state={state!r}\n  → {description}"
