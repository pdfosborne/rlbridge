"""
RL agent save/load MCP tools for the rlbridge plugin.

Tools: rl_list_trained_agents, rl_load_agent

Internal helpers: _copy_if_exists, _package_trained_agent
"""

from __future__ import annotations

import inspect
import json
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ._state import (
    _cache_root,
    _custom_env_cache_root,
    _env_agents_dir,
    _env_cache_dir,
    _safe_env_name,
    _custom_translators,
    _trained_agents,
    mcp,
)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _package_trained_agent(agent_id: str, entry: dict[str, Any]) -> tuple[Path | None, str | None]:
    """Persist agent weights plus reproducibility artifacts and create a zip bundle."""
    env_id = str(entry.get("env_id", ""))
    if not env_id:
        return None, "missing env_id"

    agent_type = str(entry.get("agent_type", "agent"))
    pkg_root = _env_agents_dir(env_id)
    pkg_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = pkg_root / f"{agent_type}_{agent_id}_{ts}"
    weights_dir = package_dir / "weights"
    env_src_dir = package_dir / "environment_source"
    translator_src_dir = package_dir / "language_translation_source"
    cached_env_root = _env_cache_dir(env_id)
    cached_env_src_dir = cached_env_root / "environment_source"
    cached_translator_src_dir = cached_env_root / "language_translation_source"
    weights_dir.mkdir(parents=True, exist_ok=True)
    env_src_dir.mkdir(parents=True, exist_ok=True)
    translator_src_dir.mkdir(parents=True, exist_ok=True)
    cached_env_src_dir.mkdir(parents=True, exist_ok=True)
    cached_translator_src_dir.mkdir(parents=True, exist_ok=True)

    agent = entry.get("agent")
    weights_path = weights_dir / "agent_weights.json"
    if agent is None or not hasattr(agent, "save"):
        return None, "agent has no save() method"

    try:
        agent.save(weights_path)
    except Exception as exc:
        return None, f"failed to save agent weights: {exc}"

    try:
        from ..environments.registry import registry as _env_registry  # noqa: PLC0415

        factory = _env_registry.get(env_id)
        factory_file = inspect.getsourcefile(type(factory))
        if factory_file:
            filename = Path(factory_file).name
            _copy_if_exists(Path(factory_file), env_src_dir / filename)
            _copy_if_exists(Path(factory_file), cached_env_src_dir / filename)

        custom_env_dir = _custom_env_cache_root() / env_id
        if custom_env_dir.exists() and custom_env_dir.is_dir():
            cached_dest = env_src_dir / "custom_env_cache"
            shutil.copytree(custom_env_dir, cached_dest, dirs_exist_ok=True)
            # Exclude the "cache" subdirectory to prevent copying a directory
            # into one of its own subdirectories, which would create an infinite
            # directory cycle.
            shutil.copytree(
                custom_env_dir,
                cached_env_src_dir / "custom_env_cache",
                ignore=shutil.ignore_patterns("cache"),
                dirs_exist_ok=True,
            )
    except Exception:
        pass

    try:
        from ..language_translation import get_translator  # noqa: PLC0415

        translator = _custom_translators.get(env_id) or get_translator(env_id)
        if translator is not None:
            translator_file = inspect.getsourcefile(type(translator))
            if translator_file:
                filename = Path(translator_file).name
                _copy_if_exists(Path(translator_file), translator_src_dir / filename)
                _copy_if_exists(Path(translator_file), cached_translator_src_dir / filename)

        custom_translator = _custom_env_cache_root() / env_id / "translator.py"
        if custom_translator.exists():
            _copy_if_exists(custom_translator, translator_src_dir / "translator.py")
            _copy_if_exists(custom_translator, cached_translator_src_dir / "translator.py")
    except Exception:
        pass

    instructions_used = list(entry.get("instructions_used") or [])
    metadata = {
        "agent_id": agent_id,
        "agent_type": agent_type,
        "env_id": env_id,
        "created_at": datetime.now().isoformat(),
        "weights_file": str(weights_path.name),
        "training_config": entry.get("training_config") or {},
        "use_language_state": bool(entry.get("use_language_state", False)),
        "match_id": entry.get("match_id"),
        "instruction": entry.get("instruction"),
        "original_instruction": entry.get("original_instruction"),
        "instructions_used": instructions_used,
        "sub_goal_language": entry.get("sub_goal_language"),
        "sub_goal_bonus": entry.get("sub_goal_bonus"),
        "sub_goal_threshold": entry.get("sub_goal_threshold"),
    }
    (package_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (package_dir / "instructions.json").write_text(
        json.dumps({"instructions": instructions_used}, indent=2),
        encoding="utf-8",
    )

    # ── Persist training history and evaluation results ────────────────────
    train_result = entry.get("train_result")
    training_results: dict = {
        "agent_name": getattr(train_result, "agent_name", agent_type) if train_result else agent_type,
        "n_episodes": getattr(train_result, "n_episodes", 0) if train_result else 0,
        "episode_rewards": list(getattr(train_result, "episode_rewards", []) if train_result else []),
        "final_epsilon": float(getattr(train_result, "final_epsilon", 0.0) if train_result else 0.0),
        "eval_mean": entry.get("eval_mean"),
        "eval_std": entry.get("eval_std"),
        "eval_rewards": list(entry.get("eval_rewards") or []),
    }
    try:
        (package_dir / "training_results.json").write_text(
            json.dumps(training_results, indent=2), encoding="utf-8"
        )
    except Exception:
        pass
    (cached_env_root / "latest_instructions.json").write_text(
        json.dumps({"instructions": instructions_used}, indent=2),
        encoding="utf-8",
    )

    archive_base = package_dir.with_suffix("")
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=package_dir)
    return Path(archive_path), None


@mcp.tool()
def rl_list_trained_agents(env_id: str = "") -> str:
    """
    List all agents that have been trained and saved, with the instruction
    each was trained on and its clean evaluation reward.

    Call this BEFORE training a new agent to check whether a suitable agent
    already exists.  If an agent was trained on the same environment and a
    matching instruction with a known eval reward, it can be reused directly
    with rl_run_agent_episode(agent_id=...) or its artifact ZIP loaded without
    re-training.

    Parameters
    ----------
    env_id:
        Restrict output to this environment.  When empty (default), all
        saved agents across every environment are shown.

    Returns
    -------
    A table of saved agents sorted by clean eval reward (best first), showing
    agent_id, type, instruction, eval reward, and artifact path.
    """
    env_base = _cache_root() / "environments"
    target = (_safe_env_name(env_id.strip()) if env_id.strip() else None)

    env_dirs: list[Path] = []
    if target:
        candidate = env_base / target
        if candidate.exists():
            env_dirs = [candidate]
    elif env_base.exists():
        env_dirs = sorted(env_base.iterdir())

    all_agents: list[dict] = []
    for env_dir in env_dirs:
        reg_path = env_dir / "agents_registry.json"
        if not reg_path.exists():
            continue
        try:
            data = json.loads(reg_path.read_text(encoding="utf-8"))
            for entry in data.values():
                if isinstance(entry, dict):
                    all_agents.append(entry)
        except Exception:
            continue

    if not all_agents:
        scope = f"'{env_id.strip()}'" if env_id.strip() else "any environment"
        return (
            f"No saved trained agents found for {scope}.\n\n"
            "Train one with rl_train_agent() and it will be registered automatically."
        )

    # Sort by eval_reward descending (None last)
    all_agents.sort(
        key=lambda x: (x.get("eval_reward") is None, -(x.get("eval_reward") or 0.0))
    )

    lines = [f"Saved trained agents - {len(all_agents)} total\n"]
    for e in all_agents:
        if e.get("eval_reward") is not None:
            eval_str = f"{e['eval_reward']:+.4f}"
            if e.get("eval_std") is not None:
                eval_str += f" ±{e['eval_std']:.4f}"
            n_eval = e.get("eval_n_episodes") or 100
            eval_str += f"  (mean±std, {n_eval} eps)"
        else:
            eval_str = "—"
        best_str  = f"{e['training_best_reward']:+.4f}" if e.get("training_best_reward") is not None else "—"
        sim_str   = f"{e['match_similarity'] * 100:.1f}%" if e.get("match_similarity") else "—"
        instr     = e.get("instruction") or "— (no instruction)"
        instr_s   = instr[:72] + ("…" if len(instr) > 72 else "")
        sub_steps = e.get("sub_steps") or []
        matched   = e.get("matched_language") or ""

        block = (
            f"  agent_id:    {e.get('agent_id', '?')}\n"
            f"  env:         {e.get('env_id', '?')}  type: {e.get('agent_type', '?')}  "
            f"episodes: {e.get('n_episodes', '?')}\n"
            f"  instruction: {instr_s}\n"
            f"  eval_reward: {eval_str}  (training best: {best_str})  sim: {sim_str}\n"
        )
        if matched:
            block += f"  matched obs: {matched[:80]}\n"
        if sub_steps:
            block += f"  sub_steps ({len(sub_steps)}): {sub_steps[0][:60]}{'…' if len(sub_steps[0]) > 60 else ''}\n"
        block += f"  saved:       {str(e.get('created_at', '?'))[:19]}\n"
        if e.get("artifact_path"):
            block += f"  archive:     {e['artifact_path']}\n"
        lines.append(block)

    lines.append(
        "To reuse a saved agent in this session: load its artifact ZIP or use\n"
        "rl_run_agent_episode(agent_id=...) if training happened in this session."
    )
    return "\n".join(lines)


@mcp.tool()
def rl_load_agent(artifact_path: str = "", agent_id: str = "") -> str:
    """
    Load a previously saved RL agent from disk into this session.

    Accepts either:
    • An artifact ZIP path (as returned by rl_train_agent or shown in
      rl_list_trained_agents) - the path to a ``<agent_type>_<id>_<ts>.zip``
      file or the unzipped package directory containing weights/ and
      metadata.json.
    • An agent_id from the persistent agents registry - the registry entry
      holds the artifact_path, so you only need to pass the agent_id you saw
      in rl_list_trained_agents.

    After loading, the agent is available in this session under the same
    agent_id that was used when it was originally trained, and can be used
    with rl_run_agent_episode(), rl_render_policy(), and
    rl_create_training_report() immediately.

    Parameters
    ----------
    artifact_path:
        Absolute or relative path to the agent's ZIP archive or the unzipped
        package directory.  Takes precedence over agent_id if both are given.
    agent_id:
        An agent_id from rl_list_trained_agents().  The registry will be
        searched across all environments to find the matching artifact path.

    Returns
    -------
    Confirmation including the loaded agent_id, type, environment, and
    the original instruction (if any).
    """
    from ..rl_agents import TabularQAgent, DQNAgent, PPOAgent  # noqa: PLC0415

    # ── Resolve the artifact path ─────────────────────────────────────────────
    resolved_path: Path | None = None

    if artifact_path.strip():
        resolved_path = Path(artifact_path.strip())
    elif agent_id.strip():
        # Search the persistent agents registry for the artifact path
        aid = agent_id.strip()
        found = False
        try:
            env_root = _cache_root() / "environments"
            if env_root.exists():
                for reg_file in env_root.glob("*/agents_registry.json"):
                    try:
                        reg_data = json.loads(reg_file.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if aid in reg_data:
                        ap = reg_data[aid].get("artifact_path") or ""
                        if ap:
                            resolved_path = Path(ap)
                            found = True
                            break
        except Exception:
            pass
        if not found or resolved_path is None:
            return (
                f"Agent ID '{aid}' not found in the persistent registry.\n"
                "Run rl_list_trained_agents() to see available saved agents."
            )
    else:
        return "Provide either artifact_path or agent_id."

    if not resolved_path.exists():
        return f"Path does not exist: {resolved_path}"

    # ── Extract ZIP if needed ─────────────────────────────────────────────────
    extract_dir: Path | None = None
    pkg_dir = resolved_path

    if resolved_path.suffix.lower() == ".zip":
        extract_dir = Path(tempfile.mkdtemp(prefix="rlip_agent_"))
        try:
            with zipfile.ZipFile(resolved_path, "r") as zf:
                zf.extractall(extract_dir)
        except Exception as exc:
            return f"Failed to extract ZIP '{resolved_path}': {exc}"
        pkg_dir = extract_dir

    # ── Read metadata ─────────────────────────────────────────────────────────
    meta_path = pkg_dir / "metadata.json"
    if not meta_path.exists():
        # Some packages wrap contents in a subdirectory
        subdirs = [d for d in pkg_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            pkg_dir = subdirs[0]
            meta_path = pkg_dir / "metadata.json"

    if not meta_path.exists():
        return f"Cannot find metadata.json inside '{resolved_path}'."

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Failed to read metadata.json: {exc}"

    loaded_agent_id = metadata.get("agent_id") or uuid.uuid4().hex[:12]
    agent_type      = metadata.get("agent_type", "").lower()
    env_id          = metadata.get("env_id", "")
    use_lang_state  = bool(metadata.get("use_language_state", False))
    instruction     = metadata.get("original_instruction") or metadata.get("instruction") or ""
    training_config = metadata.get("training_config") or {}

    # ── Locate and load weights ───────────────────────────────────────────────
    weights_file = metadata.get("weights_file", "agent_weights.json")
    weights_path = pkg_dir / "weights" / weights_file
    if not weights_path.exists():
        # Fallback: search recursively
        candidates = list(pkg_dir.rglob(weights_file))
        if not candidates:
            return f"Cannot find weights file '{weights_file}' inside the package."
        weights_path = candidates[0]

    # Instantiate agent with saved hyper-parameters
    try:
        if agent_type == "tabular_q":
            agent = TabularQAgent(n_actions=2)
        elif agent_type == "dqn":
            agent = DQNAgent(
                hidden_size=training_config.get("hidden_size") or 64,
            )
        elif agent_type == "ppo":
            agent = PPOAgent(
                hidden_size=training_config.get("hidden_size") or 64,
            )
        else:
            return (
                f"Unknown agent type '{agent_type}' in metadata.  "
                "Cannot instantiate agent."
            )
        agent.load(weights_path)
    except Exception as exc:
        return f"Failed to load agent weights: {exc}"

    # ── Restore training history and evaluation results ─────────────────────
    from .._agent_base import TrainResult  # noqa: PLC0415

    restored_train_result = None
    restored_eval_mean: float | None = None
    restored_eval_std: float | None = None
    restored_eval_rewards: list[float] = []

    tr_path = pkg_dir / "training_results.json"
    if tr_path.exists():
        try:
            tr_data = json.loads(tr_path.read_text(encoding="utf-8"))
            restored_train_result = TrainResult(
                agent_name=str(tr_data.get("agent_name") or agent_type),
                n_episodes=int(tr_data.get("n_episodes") or 0),
                episode_rewards=[float(r) for r in tr_data.get("episode_rewards") or []],
                final_epsilon=float(tr_data.get("final_epsilon") or 0.0),
            )
            if tr_data.get("eval_mean") is not None:
                restored_eval_mean = float(tr_data["eval_mean"])
            if tr_data.get("eval_std") is not None:
                restored_eval_std = float(tr_data["eval_std"])
            restored_eval_rewards = [float(r) for r in tr_data.get("eval_rewards") or []]
        except Exception:
            pass

    # ── Register in session cache ─────────────────────────────────────────────
    _trained_agents[loaded_agent_id] = {
        "agent":                agent,
        "env_id":               env_id,
        "agent_type":           agent_type,
        "use_language_state":   use_lang_state,
        "train_result":         restored_train_result,
        "best_episode_history": [],
        "training_config":      training_config,
        "match_id":             metadata.get("match_id"),
        "instruction":          metadata.get("instruction"),
        "original_instruction": instruction,
        "sub_goal_language":    metadata.get("sub_goal_language"),
        "sub_goal_bonus":       metadata.get("sub_goal_bonus"),
        "sub_goal_threshold":   metadata.get("sub_goal_threshold"),
        "eval_mean":            restored_eval_mean,
        "eval_std":             restored_eval_std,
        "eval_rewards":         restored_eval_rewards,
        "artifact_archive":     str(resolved_path) if resolved_path.suffix.lower() == ".zip" else None,
        "loaded_from":          str(resolved_path),
    }

    instr_line = f"\n  Instruction:  {instruction!r}" if instruction else ""
    lang_line  = "\n  Language obs: ON" if use_lang_state else ""

    history_lines: list[str] = []
    if restored_train_result is not None and restored_train_result.episode_rewards:
        history_lines.append(
            f"\n  Training history restored: {len(restored_train_result.episode_rewards)} episodes"
        )
    if restored_eval_mean is not None:
        eval_summary = f"\n  Eval results restored:    mean={restored_eval_mean:.4f}"
        if restored_eval_std is not None:
            eval_summary += f" ±{restored_eval_std:.4f}"
        if restored_eval_rewards:
            eval_summary += f" (n={len(restored_eval_rewards)} episodes)"
        history_lines.append(eval_summary)

    return (
        f"Agent loaded successfully.\n\n"
        f"  Agent ID:    {loaded_agent_id}\n"
        f"  Type:        {agent_type}\n"
        f"  Environment: {env_id}"
        + instr_line
        + lang_line
        + "".join(history_lines)
        + f"\n\nThe agent is ready to use:\n"
        f"  rl_run_agent_episode(agent_id='{loaded_agent_id}')\n"
        f"  rl_render_policy(env_id='{env_id}', agent_id='{loaded_agent_id}')\n"
        f"  rl_evaluate_agent(agent_id='{loaded_agent_id}')\n"
        f"  rl_create_training_report(agent_id='{loaded_agent_id}')"
    )
