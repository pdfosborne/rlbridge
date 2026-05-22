"""
RL agent evaluation MCP tools for the rlbridge plugin.

Tools: rl_run_agent_episode, rl_create_training_report, rl_evaluate_agent
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Optional

from ._dashboard import publish_report_result as _publish_report_result
from ._env_wrappers import _LangStateEnv
from ._state import (
    _env_reports_dir,
    _custom_translators,
    _instruction_protocols,
    _trained_agents,
    mcp,
)

@mcp.tool()
def rl_run_agent_episode(
    agent_id: str,
    max_steps: int = 200,
    seed: Optional[int] = None,
    stochastic: bool = False,
    use_language_state: bool = False,
) -> str:
    """
    Run a single evaluation episode using a previously trained agent.

    The agent acts greedily by default (deterministic best action).  Set
    stochastic=True for PPO's sampled-action mode.

    Parameters
    ----------
    agent_id:
        The agent_id returned by rl_train_agent().
    max_steps:
        Maximum steps for the evaluation episode.
    seed:
        Optional seed for the environment reset.
    stochastic:
        If True, use stochastic (sampled) action selection - meaningful for
        PPO; equivalent to greedy for tabular_q and dqn.
    use_language_state:
        When True, observations are translated to natural-language strings
        before being fed to the agent, matching the training mode used when
        the agent was trained with use_language_state=True.  If omitted,
        defaults to whatever value was used during training.

    Returns
    -------
    Episode summary including total reward, number of steps, end reason, and
    a trajectory excerpt.
    """
    from ..environments.registry import registry as _env_registry  # noqa: PLC0415

    entry = _trained_agents.get(agent_id)
    if entry is None:
        return (
            f"Agent ID '{agent_id}' not found.  "
            "Run rl_train_agent() first to train an agent."
        )

    agent      = entry["agent"]
    env_id     = entry["env_id"]
    agent_type = entry["agent_type"]
    # Honour the training-time language-state flag unless caller overrides
    effective_lang_state = use_language_state or entry.get("use_language_state", False)

    try:
        factory = _env_registry.get(env_id)
        env = factory.create()
    except KeyError:
        return f"Environment '{env_id}' is no longer registered."

    # ── Apply language-state wrapper if requested ─────────────────────────────
    if effective_lang_state:
        from ..language_translation import get_translator  # noqa: PLC0415
        translator = _custom_translators.get(env_id) or get_translator(env_id)
        if translator is None:
            return (
                f"use_language_state=True requires a language translator for "
                f"'{env_id}', but none is registered.\n"
                "Register one with rl_set_translator_code() first."
            )
        env = _LangStateEnv(env, translator=translator, env_id=env_id)

    reset_out = env.reset(seed=seed)
    obs = reset_out.observation if hasattr(reset_out, "observation") else reset_out.get("observation", reset_out)

    total_reward = 0.0
    step_count   = 0
    trajectory:  list[str] = []
    terminated   = False
    truncated    = False

    for step_n in range(1, max_steps + 1):
        if stochastic and hasattr(agent, "act"):
            # PPOAgent has act() (stochastic) and act_greedy()
            action = agent.act(obs)
        else:
            act_fn = getattr(agent, "act_greedy", None) or agent.act
            action = act_fn(obs)

        step_out = env.step(action)
        obs        = step_out.observation if hasattr(step_out, "observation") else step_out.get("observation", obs)
        reward     = float(step_out.reward if hasattr(step_out, "reward") else step_out.get("reward", 0.0))
        terminated = bool(step_out.terminated if hasattr(step_out, "terminated") else step_out.get("terminated", False))
        truncated  = bool(step_out.truncated if hasattr(step_out, "truncated") else step_out.get("truncated", False))

        total_reward += reward
        step_count   = step_n

        if step_n <= 10:
            trajectory.append(f"  step {step_n:3d}  action={action}  r={reward:+.3f}  obs={str(obs)[:60]}")

        if terminated or truncated:
            break

    if step_count > 10:
        trajectory.append(f"  … ({step_count - 10} more steps not shown)")

    end_reason = "terminated" if terminated else ("truncated" if truncated else "max_steps")
    mode_str   = "stochastic" if stochastic else "greedy"
    lang_mode  = "language" if effective_lang_state else "raw"

    return (
        f"Evaluation episode - {agent_type} on {env_id}\n"
        f"  Agent ID:     {agent_id}\n"
        f"  Action mode:  {mode_str}\n"
        f"  Obs mode:     {lang_mode}\n"
        f"  Steps:        {step_count}\n"
        f"  Total reward: {total_reward:.4f}\n"
        f"  End reason:   {end_reason}\n\n"
        "Trajectory:\n" + "\n".join(trajectory)
    )


def _run_clean_evaluation(
    agent: Any,
    env_factory: Any,
    n_episodes: int = 100,
    max_steps: int = 200,
    use_language_state: bool = False,
    translator: Any = None,
    env_id: str = "",
    seed: int = 0,
) -> tuple[float, float, list[float]]:
    """
    Evaluate *agent* with fixed weights on the plain environment for *n_episodes*.

    The environment is created fresh with no instruction/shaping wrappers so
    the reported reward reflects only the true environment signal.  If the
    agent was trained with language-state observations the same
    ``_LangStateEnv`` wrapper is applied so the observation format matches,
    but no bonus rewards are added.

    Returns
    -------
    (mean_reward, std_reward, episode_rewards)
    """
    from ..interaction_protocols import GreedyEpisodeProtocol, MultiEpisodeProtocol  # noqa: PLC0415

    eval_env = env_factory.create(render_mode=None)
    try:
        if use_language_state and translator is not None:
            eval_env = _LangStateEnv(eval_env, translator=translator, env_id=env_id)

        def _greedy_fn(obs: Any) -> Any:
            if hasattr(agent, "act_greedy"):
                return agent.act_greedy(obs)
            return agent.act(obs)

        protocol = MultiEpisodeProtocol(
            GreedyEpisodeProtocol(
                policy_fn=_greedy_fn,
                max_steps=max_steps,
                seed=seed,
                record_history=False,
            ),
            n_episodes=n_episodes,
            base_seed=seed,
        )
        result = protocol(eval_env)
    finally:
        eval_env.close()

    rewards = [ep.total_reward for ep in result.episodes]
    if not rewards:
        return 0.0, 0.0, []
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = variance ** 0.5
    return mean, std, rewards


@mcp.tool()
def rl_create_training_report(
    agent_id: str,
    compare_agent_ids: Optional[list[str]] = None,
    output_path: str = "",
    rolling_window: int = 0,
    breakpoint_count: int = 6,
    n_sample_frames: int = 6,
    render_for_frames: bool = True,
    n_render_episodes: int = 20,
    fps: float = 6.0,
) -> str:
    """
     Generate a redesigned training report for one or more trained agents and
     save it as a PNG image.

     The report includes:
     1. Reward-convergence evaluation from training reward trajectories.
     2. Reward obtained by the optimal policy at training breakpoints
         (best-so-far proxy: max reward observed up to each breakpoint).
     3. Instructions used, best-matched observation, and similarity %.
     4. Metadata and hyper-parameters used for each compared agent.

    Parameters
    ----------
    agent_id:
        Primary agent_id returned by rl_train_agent().
    compare_agent_ids:
        Optional additional agent IDs to compare in the same report.
        Agents should be trained on the same environment for fair comparison.
    output_path:
        Path to write the PNG report.  Defaults to
        ``<cwd>/rlbridge_results/<env>/reports/<env>_<agent_id>_report.png``.
    rolling_window:
        Number of episodes for the rolling reward average (0 = auto: 5 %
        of total episodes, minimum 10).
    breakpoint_count:
        Number of training breakpoints used in the optimal-policy evaluation.
    n_sample_frames:
        Deprecated. Kept for backward compatibility.
    render_for_frames:
        Deprecated. Kept for backward compatibility.
    n_render_episodes:
        Deprecated. Kept for backward compatibility.
    fps:
        Deprecated. Kept for backward compatibility.

    Returns
    -------
    The local path of the saved PNG report and its base64-encoded content
    so Claude can display it inline.
    """
    del n_sample_frames
    del render_for_frames
    del n_render_episodes
    del fps

    from ..analysis import create_training_report  # noqa: PLC0415

    def _breakpoint_proxy(
        rewards: list[float],
        count: int,
    ) -> list[tuple[int, float]]:
        if not rewards:
            return []
        n = len(rewards)
        del count

        # Phase 1: every 10 episodes for the first 100 episodes
        episodes: list[int] = list(range(10, min(101, n + 1), 10))
        
        # Phase 2: larger intervals for episodes > 100 (every 50 episodes)
        if n > 100:
            episodes.extend(ep for ep in range(150, n + 1, 50))
            if n not in episodes:
                episodes.append(n)
        
        episodes = sorted(set(episodes))  # Remove duplicates and sort

        out: list[tuple[int, float]] = []
        running_best = float("-inf")
        next_idx = 0
        for ep_idx, reward in enumerate(rewards, start=1):
            running_best = max(running_best, reward)
            while next_idx < len(episodes) and ep_idx >= episodes[next_idx]:
                out.append((episodes[next_idx], running_best))
                next_idx += 1
        return out

    compare_agent_ids = compare_agent_ids or []
    requested_ids: list[str] = []
    for cand in [agent_id, *compare_agent_ids]:
        if cand and cand not in requested_ids:
            requested_ids.append(cand)

    missing = [aid for aid in requested_ids if aid not in _trained_agents]
    if missing:
        return (
            "Some agent IDs were not found: "
            + ", ".join(missing)
            + ". Run rl_train_agent() first."
        )

    entries = [(aid, _trained_agents[aid]) for aid in requested_ids]
    env_ids = {entry["env_id"] for _, entry in entries}
    if len(env_ids) != 1:
        return (
            "All compared agents must come from the same environment. "
            f"Found environments: {', '.join(sorted(env_ids))}"
        )

    env_id = entries[0][1]["env_id"]
    primary_agent_type = entries[0][1]["agent_type"]

    for aid, entry in entries:
        if entry.get("train_result") is None:
            return (
                f"Agent '{aid}' has no stored training result. "
                "Re-train that agent to enable reporting."
            )

    env_reports_dir = _env_reports_dir(env_id)
    env_reports_dir.mkdir(parents=True, exist_ok=True)
    safe_id = env_id.replace("/", "_").replace("-", "_").replace(" ", "_")
    if len(entries) > 1:
        out_stem = output_path or str(env_reports_dir / f"{safe_id}_comparison_report")
    else:
        out_stem = output_path or str(
            env_reports_dir / f"{safe_id}_{primary_agent_type}_{agent_id}_report"
        )
    # Strip .png suffix if caller passed a full path - we derive per-file suffixes
    if out_stem.endswith(".png"):
        out_stem = out_stem[:-4]

    comparison_runs: list[dict[str, Any]] = []
    for aid, entry in entries:
        tr = entry["train_result"]
        agent_type = entry["agent_type"]

        hp_raw = dict(entry.get("training_config") or {})
        hp = {k: v for k, v in hp_raw.items() if v is not None}

        uses_instructions = bool(entry.get("instruction") or entry.get("match_id"))
        md: dict[str, Any] = {
            "env_id": entry.get("env_id"),
            "agent_id": aid,
            "agent_type": agent_type,
            "use_language_state": bool(entry.get("use_language_state", False)),
            "uses_instructions": uses_instructions,
            "match_id": entry.get("match_id") or "-",
        }

        # Build per-sub-step match info for the instructions figure.
        # Each sub-step is individually matched to a language-translated env
        # observation; we record the matched language (= text render of that
        # state) and similarity so the report can show them side-by-side.
        sub_steps_matches: list[dict[str, Any]] = []
        match_id = entry.get("match_id")
        if match_id and match_id in _instruction_protocols:
            proto_entry = _instruction_protocols[match_id]
            proto = proto_entry.get("protocol")
            instructions_list: list[str] = list(proto_entry.get("instructions") or [])
            per_step_matches = list(getattr(proto, "matches", None) or [])
            for step_instr, m in zip(instructions_list, per_step_matches):
                sub_steps_matches.append({
                    "instruction":       step_instr,
                    "matched_language":  getattr(m, "matched_language", "") or "—",
                    "similarity":        getattr(m, "similarity_score", None),
                    # render_text: the language description IS the text render of
                    # the matched observation for all rlbridge text environments.
                    "render_text":       getattr(m, "matched_language", "") or None,
                    "render_b64":        None,
                })

        comparison_runs.append(
            {
                "label": f"{agent_type}:{aid[:6]}",
                "train_result": tr,
                "metadata": md,
                "hyperparameters": hp,
                "instruction": entry.get("instruction"),
                "best_match_observation": entry.get("best_match_observation") or entry.get("sub_goal_language"),
                "best_match_similarity": entry.get("best_match_similarity"),
                "breakpoints": _breakpoint_proxy(tr.episode_rewards, breakpoint_count),
                "eval_mean": entry.get("eval_mean"),
                "eval_std": entry.get("eval_std"),
                "eval_rewards": list(entry.get("eval_rewards") or []),
                "sub_steps_matches": sub_steps_matches,
            }
        )

    try:
        figs = create_training_report(
            comparison_runs=comparison_runs,
            output_path=out_stem,   # analysis.py now appends _rewards/_instructions/_config
            rolling_window=rolling_window or None,
            n_breakpoints=breakpoint_count,
        )
    except Exception as exc:
        return f"Report generation failed: {exc}"

    # Encode all three figures as base64 so the client can display them inline
    import io as _io  # noqa: PLC0415

    encoded: list[str] = []
    encoded_by_key: dict[str, str] = {}
    saved_paths: list[str] = []
    for key, suffix in [("rewards", "_rewards.png"), ("instructions", "_instructions.png"), ("config", "_config.png")]:
        fig = figs.get(key)
        if fig is None:
            continue
        saved_paths.append(out_stem + suffix)
        try:
            buf = _io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("ascii")
            encoded_by_key[key] = b64
            encoded.append(f"### {key.capitalize()} chart\ndata:image/png;base64,{b64}")
        except Exception as exc:
            encoded.append(f"### {key.capitalize()} chart\n[encoding failed: {exc}]")

    if encoded_by_key:
        try:
            _publish_report_result(
                out_stem,
                env_id=env_id,
                agent_ids=requested_ids,
                generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                images=encoded_by_key,
            )
        except Exception:
            pass

    header = (
        f"Training report generated for {len(entries)} agent(s) on {env_id}.\n"
        f"Agent IDs: {', '.join(requested_ids)}\n"
        f"Saved to:\n" + "\n".join(f"  {p}" for p in saved_paths) + "\n\n"
    )
    return header + "\n\n".join(encoded)


@mcp.tool()
def rl_evaluate_agent(
    agent_id: str,
    n_episodes: int = 100,
    max_steps: int = 200,
    seed: int = 0,
    force_rerun: bool = False,
) -> str:
    """
    Run a clean evaluation of a trained agent and return standard metrics.

    Evaluation procedure:
    - Load the trained agent's weights (fixed - no further learning).
    - Create a fresh environment instance with **no** instruction rewards or
      shaping wrappers; only the plain environment reward signal is used.
    - If the agent was trained with language-state observations the same
      language-state wrapper is applied so observation formats match, but
      no bonus rewards are injected.
    - Run for *n_episodes* episodes and collect per-episode rewards.

    Metrics returned
    ----------------
    - mean reward ± std (primary comparison metric)
    - min / max episode reward
    - median reward
    - 25th / 75th percentile
    - number of episodes
    - fraction of episodes where reward > 0  (success proxy)
    - training statistics for reference (best, last-10% mean)

    If a clean evaluation was already run after training (stored on the agent)
    those cached results are returned immediately unless *force_rerun* is True.

    Parameters
    ----------
    agent_id:
        Agent ID returned by rl_train_agent() or rl_load_agent().
    n_episodes:
        Number of evaluation episodes (default 100).
    max_steps:
        Maximum steps per episode (should match training setting).
    seed:
        Base seed for reproducibility across episodes.
    force_rerun:
        When True, ignore any cached evaluation and re-run from scratch.

    Returns
    -------
    A structured text report of all evaluation metrics.
    """
    import math  # noqa: PLC0415

    entry = _trained_agents.get(agent_id)
    if entry is None:
        return (
            f"Agent '{agent_id}' not found in this session.\n"
            "Use rl_list_trained_agents() to see saved agents or rl_load_agent() "
            "to load one from disk."
        )

    env_id = str(entry.get("env_id", ""))
    agent_type = str(entry.get("agent_type", "?"))
    use_language_state = bool(entry.get("use_language_state", False))
    agent = entry.get("agent")
    if agent is None:
        return f"Agent '{agent_id}' has no loaded weights in this session."

    # ── Use cached results unless force_rerun ────────────────────────────────
    cached_mean = entry.get("eval_mean")
    cached_std = entry.get("eval_std")
    cached_rewards: list[float] = list(entry.get("eval_rewards") or [])

    if (
        not force_rerun
        and cached_mean is not None
        and cached_std is not None
        and len(cached_rewards) > 0
    ):
        rewards = cached_rewards
        source = "cached (from post-training evaluation)"
    else:
        # ── Run fresh clean evaluation ───────────────────────────────────────
        try:
            from ..environments.registry import registry as _eval_env_registry  # noqa: PLC0415
            from ..language_translation import get_translator as _get_translator  # noqa: PLC0415

            factory = _eval_env_registry.get(env_id)
            translator = (
                _custom_translators.get(env_id) or _get_translator(env_id)
                if use_language_state else None
            )
            _, _, rewards = _run_clean_evaluation(
                agent=agent,
                env_factory=factory,
                n_episodes=n_episodes,
                max_steps=max_steps,
                use_language_state=use_language_state,
                translator=translator,
                env_id=env_id,
                seed=seed,
            )
        except Exception as exc:
            return f"Evaluation failed: {exc}"

        # Store for future calls
        if rewards:
            mean = sum(rewards) / len(rewards)
            variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
            std = variance ** 0.5
            entry["eval_mean"] = mean
            entry["eval_std"] = std
            entry["eval_rewards"] = rewards

        source = f"fresh run ({n_episodes} episodes requested, {len(rewards)} completed)"

    if not rewards:
        return "Evaluation produced no completed episodes - check environment and agent compatibility."

    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = variance ** 0.5
    sorted_r = sorted(rewards)
    minimum = sorted_r[0]
    maximum = sorted_r[-1]
    median = sorted_r[n // 2] if n % 2 == 1 else (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2.0
    q1 = sorted_r[n // 4]
    q3 = sorted_r[(3 * n) // 4]
    positive_frac = sum(1 for r in rewards if r > 0) / n
    negative_frac = sum(1 for r in rewards if r < 0) / n
    zero_frac = 1.0 - positive_frac - negative_frac

    # Training reference stats
    tr = entry.get("train_result")
    training_lines = ""
    if tr is not None:
        training_lines = (
            f"\n  Training reference:\n"
            f"    episodes trained : {tr.n_episodes}\n"
            f"    training best    : {tr.best_reward:.4f}\n"
            f"    training last10% : {tr.last_n_mean:.4f}\n"
        )

    instruction = entry.get("instruction") or entry.get("original_instruction") or ""
    instruction_line = f"    instruction      : {instruction[:80]}\n" if instruction else ""

    return (
        f"Clean evaluation - {agent_type} on {env_id}\n"
        f"  agent_id : {agent_id}\n"
        f"  source   : {source}\n"
        f"{instruction_line}"
        f"\n"
        f"  Episodes          : {n}\n"
        f"  Mean reward       : {mean:.4f}\n"
        f"  Std               : {std:.4f}\n"
        f"  Min               : {minimum:.4f}\n"
        f"  Max               : {maximum:.4f}\n"
        f"  Median            : {median:.4f}\n"
        f"  25th percentile   : {q1:.4f}\n"
        f"  75th percentile   : {q3:.4f}\n"
        f"  Episodes > 0      : {positive_frac * 100:.1f}%\n"
        f"  Episodes = 0      : {zero_frac * 100:.1f}%\n"
        f"  Episodes < 0      : {negative_frac * 100:.1f}%\n"
        f"{training_lines}"
        f"\n"
        f"  Evaluation: fixed weights, no instruction rewards, plain environment.\n"
        f"  Use rl_create_training_report() to compare multiple agents visually."
    )
