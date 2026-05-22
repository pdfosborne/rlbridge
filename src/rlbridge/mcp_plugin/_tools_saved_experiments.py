"""
Saved experiment management MCP tools for the rlbridge plugin.

An "experiment" is a confirmed, named configuration that captures:
  - the user's original objective (prompt)
  - the environment, agent type, instruction, and language-translation mode
    that the LLM and user agreed best achieves that objective
  - the clean evaluation metrics for reproducibility
  - an optional conversation log (key turns that led to the decision)

Tools
-----
rl_save_experiment      - persist a confirmed configuration to cache
rl_list_experiments     - list all saved experiments (optionally by env)
rl_load_experiment      - recall a saved experiment by ID
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ._prompts import EXPERIMENT_SAVE_GUIDE
from ._state import (
    _cache_root,
    _experiments_local_dir,
    _experiments_registry_path,
    _safe_env_name,
    _saved_experiments,
    _trained_agents,
    log,
    mcp,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _persist_experiments_registry() -> None:
    """Write the in-memory ``_saved_experiments`` dict back to disk."""
    reg_path = _experiments_registry_path()
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(_saved_experiments, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def rl_save_experiment(
    user_objective: str,
    env_id: str,
    agent_id: str,
    agent_type: str,
    instruction: str = "",
    use_language_state: bool = False,
    eval_mean: Optional[float] = None,
    eval_std: Optional[float] = None,
    name: str = "",
    notes: str = "",
    save_local: bool = False,
    prompt_log: Optional[list[dict]] = None,
) -> str:
    """
    Save a confirmed experiment configuration to the rlbridge cache.

    Call this once the LLM and user have agreed on the best agent, instruction,
    and training configuration for the user's objective.  The experiment is
    stored in ``~/.rlbridge/experiments_registry.json`` and can be recalled with
    ``rl_load_experiment()`` in any future session.

    When to call this tool
    ----------------------
    After verifying the agent passes evaluation (``rl_evaluate_agent`` shows
    meaningful reward above random AND ``rl_run_agent_episode`` completes the
    goal), summarise the confirmed configuration here.  Include the original
    user objective verbatim so future sessions can match experiments to goals.

    Parameters
    ----------
    user_objective:
        The user's original goal or task description (verbatim or summarised).
        This is the primary search key when recalling experiments later.
    env_id:
        The registered rlbridge environment ID the agent was trained on.
    agent_id:
        The agent_id returned by ``rl_train_agent`` / ``rl_experiment_process``
        for the best performing agent in this session.
    agent_type:
        Agent type used: ``"tabular_q"``, ``"dqn"``, or ``"ppo"``.
    instruction:
        The natural-language instruction that shaped training (empty string if
        no instruction shaping was used).
    use_language_state:
        Whether the agent uses language-translated observations instead of raw
        numeric vectors.
    eval_mean:
        Mean episode reward from ``rl_evaluate_agent()``.  Include whenever
        available so experiments can be ranked by performance.
    eval_std:
        Standard deviation of episode reward from ``rl_evaluate_agent()``.
    name:
        Short human-readable experiment name.  Auto-generated from
        ``<env_id>_<timestamp>`` if omitted.
    notes:
        Free-text explanation from the LLM of why this configuration was
        chosen and what was tried before reaching it.  Include relevant
        observations about instruction quality, training stability, or
        environment properties.
    save_local:
        When True, an additional copy of the experiment JSON is written to
        ``<cwd>/rlbridge_results/<env_id>/experiments/`` for easy project-level
        access alongside rendered policies and training reports.
    prompt_log:
        Optional list of ``{"role": ..., "content": ...}`` dicts capturing
        the key conversation turns that led to this configuration.  Useful for
        reconstructing how the objective was explored.  Omit or pass an empty
        list to skip.

    Returns
    -------
    Confirmation string with the assigned ``experiment_id`` and save paths.
    """
    experiment_id = uuid.uuid4().hex[:12]
    ts = datetime.now().isoformat()
    exp_name = name.strip() or f"{_safe_env_name(env_id)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Pull extra metadata from the live agent entry if available
    agent_entry = _trained_agents.get(agent_id, {})
    artifact_path: str = str(agent_entry.get("artifact_path", "")) if agent_entry else ""

    record: dict[str, Any] = {
        "experiment_id": experiment_id,
        "name": exp_name,
        "created_at": ts,
        "user_objective": user_objective,
        "env_id": env_id,
        "agent_id": agent_id,
        "agent_type": agent_type,
        "instruction": instruction,
        "use_language_state": use_language_state,
        "eval_mean": eval_mean,
        "eval_std": eval_std,
        "notes": notes,
        "artifact_path": artifact_path,
        "prompt_log": prompt_log or [],
    }

    _saved_experiments[experiment_id] = record
    _persist_experiments_registry()
    saved_paths = [str(_experiments_registry_path())]

    local_path_msg = ""
    if save_local:
        local_dir = _experiments_local_dir(env_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / f"{experiment_id}.json"
        local_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
        saved_paths.append(str(local_file))
        local_path_msg = f"\n  Local copy:     {local_file}"

    eval_str = ""
    if eval_mean is not None:
        std_str = f" ± {eval_std:.4f}" if eval_std is not None else ""
        eval_str = f"\n  Eval reward:    {eval_mean:.4f}{std_str}"

    return (
        f"Experiment saved.\n\n"
        f"  Experiment ID:  {experiment_id}\n"
        f"  Name:           {exp_name}\n"
        f"  Environment:    {env_id}\n"
        f"  Agent ID:       {agent_id} ({agent_type})\n"
        f"  Instruction:    {instruction!r}\n"
        f"  Language state: {use_language_state}"
        f"{eval_str}\n"
        f"  Cache registry: {_experiments_registry_path()}"
        f"{local_path_msg}\n\n"
        f"Recall with: rl_load_experiment(experiment_id='{experiment_id}')"
    )


@mcp.tool()
def rl_list_experiments(
    env_id: str = "",
    limit: int = 20,
) -> str:
    """
    List saved experiments from the rlbridge cache, newest first.

    Call this at the start of a session to check whether a previous experiment
    already solved the user's objective.  If a matching experiment exists with
    a strong eval reward, use ``rl_load_experiment()`` to recover the
    configuration and optionally reload the agent with ``rl_load_agent()``.

    Parameters
    ----------
    env_id:
        Restrict output to experiments for this environment.  When empty
        (default), all saved experiments across every environment are shown.
    limit:
        Maximum number of experiments to display (most recent first).

    Returns
    -------
    A table of saved experiments with their IDs, names, environments,
    instructions, evaluation rewards, and objectives.
    """
    if not _saved_experiments:
        return (
            "No saved experiments found.\n"
            "Use rl_save_experiment() after validating a trained agent to "
            "store the confirmed configuration."
        )

    target = env_id.strip().lower()
    rows = [
        rec for rec in _saved_experiments.values()
        if not target or rec.get("env_id", "").lower() == target
    ]

    if not rows:
        return f"No saved experiments found for environment '{env_id}'."

    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    rows = rows[:limit]

    lines = [
        f"Saved experiments ({len(rows)} shown"
        + (f", filtered to env={env_id}" if target else "")
        + "):\n",
        f"{'ID':<14}  {'Name':<28}  {'Env':<18}  {'Agent':<10}  {'EvalMean':>9}  {'Objective':<40}",
        "-" * 130,
    ]
    for r in rows:
        ev = r.get("eval_mean")
        ev_str = f"{ev:.3f}" if ev is not None else "  n/a  "
        obj = r.get("user_objective", "")[:38] + ("…" if len(r.get("user_objective", "")) > 38 else "")
        lines.append(
            f"{r['experiment_id']:<14}  "
            f"{r.get('name', '')[:27]:<28}  "
            f"{r.get('env_id', '')[:17]:<18}  "
            f"{r.get('agent_type', ''):<10}  "
            f"{ev_str:>9}  "
            f"{obj:<40}"
        )

    lines.append(
        "\nRecall details with: rl_load_experiment(experiment_id='<id>')"
    )
    return "\n".join(lines)


@mcp.tool()
def rl_load_experiment(experiment_id: str) -> str:
    """
    Recall the full configuration of a previously saved experiment.

    Returns the experiment record including the user objective, environment,
    agent configuration, instruction, evaluation metrics, LLM notes, and
    any saved conversation log.  If the agent artifact path is available,
    you can restore it into the current session with ``rl_load_agent()``.

    Parameters
    ----------
    experiment_id:
        The ``experiment_id`` returned by ``rl_save_experiment()`` or shown
        in ``rl_list_experiments()``.

    Returns
    -------
    Full experiment details and, if applicable, instructions for restoring
    the agent.
    """
    record = _saved_experiments.get(experiment_id)
    if record is None:
        # Try reloading from disk in case the registry file was updated
        # by another session since import time.
        reg = _experiments_registry_path()
        if reg.exists():
            try:
                data = json.loads(reg.read_text(encoding="utf-8"))
                record = data.get(experiment_id)
                if record:
                    _saved_experiments.update(data)
            except Exception:
                pass
    if record is None:
        return (
            f"Experiment '{experiment_id}' not found.\n"
            "Use rl_list_experiments() to see all saved experiment IDs."
        )

    ev = record.get("eval_mean")
    ev_str = f"{ev:.4f}" if ev is not None else "n/a"
    std = record.get("eval_std")
    std_str = f" ± {std:.4f}" if std is not None else ""

    artifact = record.get("artifact_path", "")
    artifact_msg = ""
    if artifact and Path(artifact).exists():
        artifact_msg = (
            f"\n\nAgent artifact found at:\n  {artifact}\n"
            f"Restore into current session with:\n"
            f"  rl_load_agent(artifact_path='{artifact}')"
        )
    elif artifact:
        artifact_msg = (
            f"\n\n(Artifact path on record: {artifact} - file not found at "
            "current working directory; may need rl_train_agent() to recreate.)"
        )

    plog = record.get("prompt_log") or []
    plog_section = ""
    if plog:
        plog_section = "\n\nConversation log (" + str(len(plog)) + " turns):\n"
        for i, turn in enumerate(plog[:6]):
            role = turn.get("role", "?")
            content = str(turn.get("content", ""))[:200]
            plog_section += f"  [{role}] {content}\n"
        if len(plog) > 6:
            plog_section += f"  … ({len(plog) - 6} more turns not shown)\n"

    notes = record.get("notes", "")
    notes_section = f"\n\nNotes:\n  {notes}" if notes else ""

    return (
        f"Experiment: {record.get('name', experiment_id)}\n"
        f"  ID:              {experiment_id}\n"
        f"  Created:         {record.get('created_at', 'unknown')}\n\n"
        f"User objective:\n  {record.get('user_objective', '')}\n\n"
        f"Configuration:\n"
        f"  Environment:     {record.get('env_id', '')}\n"
        f"  Agent type:      {record.get('agent_type', '')}\n"
        f"  Agent ID:        {record.get('agent_id', '')}\n"
        f"  Instruction:     {record.get('instruction', '') or '(none)'}\n"
        f"  Language state:  {record.get('use_language_state', False)}\n\n"
        f"Evaluation:\n"
        f"  Mean reward:     {ev_str}{std_str}"
        f"{notes_section}"
        f"{artifact_msg}"
        f"{plog_section}"
    )
