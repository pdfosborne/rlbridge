"""
RL agent training and evaluation MCP tools for the RLIP plugin.

Tools: rl_list_agents, rl_train_agent, rl_run_agent_episode,
       rl_create_training_report.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import shutil
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import Context

from ._dashboard import dashboard as _dash, is_running as _dash_running
from ._env_wrappers import _LangStateEnv, _SequentialShapedEnv, _ShapedEnv
from ._state import (
    _cache_root,
    _custom_env_cache_root,
    _env_agents_dir,
    _env_agents_registry_path,
    _env_cache_dir,
    _env_renders_dir,
    _env_reports_dir,
    _safe_env_name,
    _custom_translators,
    _instruction_protocols,
    _trained_agents,
    _training_jobs,
    log,
    mcp,
)

class _ProgressEnv:
    """
    Thin env wrapper that renders a tqdm progress bar on sys.stderr during
    agent.train().  Intercepts reset() to advance the bar once per episode
    and step() to track the current episode reward for the postfix display.

    Written to stderr so it never touches the MCP stdout channel.
    """

    def __init__(
        self,
        env: Any,
        n_episodes: int,
        agent_type: str,
        env_id: str,
        *,
        dashboard_agent_id: str = "",
        agent_ref: Any = None,
    ) -> None:
        import tqdm
        self._env = env
        self._n_episodes = n_episodes
        self._best_reward = float("-inf")
        self._ep_reward = 0.0
        self._reset_calls = 0
        self._completed_episodes = 0
        self._dashboard_agent_id = dashboard_agent_id
        self._agent_ref = agent_ref
        self._bar = tqdm.tqdm(
            total=n_episodes,
            desc=f"Training {agent_type} on {env_id}",
            unit="ep",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
        )

    def reset(self, seed: Any = None, options: Any = None) -> Any:
        if self._reset_calls > 0:
            # A previous episode just ended — commit its reward and advance.
            if self._ep_reward > self._best_reward:
                self._best_reward = self._ep_reward
            self._bar.set_postfix(
                last=f"{self._ep_reward:.2f}",
                best=f"{self._best_reward:.2f}",
            )
            n = min(1, self._n_episodes - self._bar.n)
            if n > 0:
                self._bar.update(n)
            _finished_reward = self._ep_reward
            self._ep_reward = 0.0
            self._completed_episodes += 1
            if self._dashboard_agent_id:
                _dash.update(
                    self._dashboard_agent_id,
                    completed=self._completed_episodes,
                    last_reward=_finished_reward,
                    epsilon=float(getattr(self._agent_ref, "epsilon", 0.0)),
                )
        self._reset_calls += 1
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        try:
            r = result.reward if hasattr(result, "reward") else result.get("reward", 0.0)
            self._ep_reward += float(r)
        except Exception:
            pass
        return result

    def close(self) -> None:
        self._bar.close()
        if self._dashboard_agent_id and self._reset_calls > 0:
            # Commit the last episode whose reward was accumulated but never
            # flushed (there is no subsequent reset() call after the final ep).
            self._completed_episodes += 1
            _dash.update(
                self._dashboard_agent_id,
                completed=min(self._completed_episodes, self._n_episodes),
                last_reward=self._ep_reward,
                epsilon=float(getattr(self._agent_ref, "epsilon", 0.0)),
            )
            _dash.finish(self._dashboard_agent_id)
        elif self._dashboard_agent_id:
            _dash.finish(self._dashboard_agent_id)
        self._env.close()

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def env_id(self) -> str:
        return getattr(self._env, "env_id", "")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)



# ── Artifact packaging helpers ───────────────────────────────────────────────

def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _collect_training_instructions(match_id: str) -> list[str]:
    if not match_id:
        return []
    entry = _instruction_protocols.get(match_id)
    if not entry:
        return []
    protocol = entry.get("protocol")
    if protocol is None:
        return []

    seq = getattr(protocol, "instructions", None)
    if isinstance(seq, list) and seq:
        return [str(x) for x in seq if str(x).strip()]

    single = getattr(protocol, "instruction", None)
    if isinstance(single, str) and single.strip():
        return [single]

    return []


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
            shutil.copytree(custom_env_dir, cached_env_src_dir / "custom_env_cache", dirs_exist_ok=True)
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
    (cached_env_root / "latest_instructions.json").write_text(
        json.dumps({"instructions": instructions_used}, indent=2),
        encoding="utf-8",
    )

    archive_base = package_dir.with_suffix("")
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=package_dir)
    return Path(archive_path), None


# ── Clean post-training evaluation ───────────────────────────────────────────

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


# ── Dashboard policy render helper ───────────────────────────────────────────

def _render_policy_for_dashboard(
    agent_id: str,
    env_id: str,
    best_episode_history: list,
    max_steps: int = 200,
) -> None:
    """
    Render the final trained agent policy and push the result to the dashboard.

    This evaluates the *final* agent greedily after training (not an early
    checkpoint history), then renders the optimal episode from that evaluation.
    No random fallback is allowed during replay: unseen states use the trained
    agent's greedy action.
    """
    try:
        import base64 as _b64  # noqa: PLC0415

        from ..environments.registry import registry as _env_registry  # noqa: PLC0415
        from ..interaction_protocols import (  # noqa: PLC0415
            GreedyEpisodeProtocol,
            MultiEpisodeProtocol,
        )
        from ..language_translation import get_translator  # noqa: PLC0415
        from ..policy_rendering import render_optimal_policy  # noqa: PLC0415

        del best_episode_history  # final render must come from end-of-training policy

        stored = _trained_agents.get(agent_id, {})
        agent = stored.get("agent")
        if agent is None:
            _dash.finish(agent_id, policy_text="Dashboard render error: trained agent not found in cache.")
            return

        use_lang_state = bool(stored.get("use_language_state", False))
        translator = _custom_translators.get(env_id) or get_translator(env_id)
        factory = _env_registry.get(env_id)

        if use_lang_state and translator is None:
            _dash.finish(
                agent_id,
                policy_text=(
                    f"Dashboard render error: agent '{agent_id}' requires language-state observations, "
                    f"but no translator is registered for '{env_id}'."
                ),
            )
            return

        # Evaluate the final trained policy greedily and keep full trajectories.
        eval_env = factory.create(render_mode=None)
        try:
            if use_lang_state:
                eval_env = _LangStateEnv(eval_env, translator=translator, env_id=env_id)

            def _greedy_fn(obs: Any) -> Any:
                if hasattr(agent, "act_greedy"):
                    return agent.act_greedy(obs)
                return agent.act(obs)

            eval_protocol = MultiEpisodeProtocol(
                GreedyEpisodeProtocol(
                    policy_fn=_greedy_fn,
                    max_steps=max_steps,
                    seed=0,
                    record_history=True,
                ),
                n_episodes=12,
                base_seed=0,
            )
            eval_result = eval_protocol(eval_env)
        finally:
            eval_env.close()

        if not getattr(eval_result, "episodes", None):
            _dash.finish(agent_id, policy_text="Dashboard render error: evaluation produced no episodes.")
            return

        # Deterministic non-random fallback for replay when a state wasn't
        # present in the extracted best-episode trajectory.
        def _greedy_missing_state(obs: Any) -> Any:
            if hasattr(agent, "act_greedy"):
                return agent.act_greedy(obs)
            return agent.act(obs)

        # Attempt 1: rgb_array GIF render
        rgb_error: Optional[str] = None
        try:
            render_dir = _env_renders_dir(env_id)
            render_dir.mkdir(parents=True, exist_ok=True)
            gif_path = render_dir / f"{agent_id}_dashboard_policy.gif"
            render_env = factory.create(render_mode="rgb_array")
            try:
                if use_lang_state:
                    render_env = _LangStateEnv(render_env, translator=translator, env_id=env_id)
                render_result = render_optimal_policy(
                    eval_result,
                    env=render_env,
                    max_steps=max_steps,
                    seed=0,
                    fallback=_greedy_missing_state,
                    translate=(translator if use_lang_state else True),
                    output_gif=gif_path,
                    gif_fps=5.0,
                    gif_annotate=True,
                )
            finally:
                render_env.close()
            if render_result.n_gif_frames > 0:
                gif_bytes = gif_path.read_bytes()
                b64 = _b64.b64encode(gif_bytes).decode("ascii")
                _dash.finish(agent_id, policy_gif_b64=b64)
                return
        except Exception as exc:
            rgb_error = str(exc)

        # Attempt 2: ANSI text frames (still deterministic, no random fallback)
        try:
            render_env = factory.create(render_mode="ansi")
            try:
                if use_lang_state:
                    render_env = _LangStateEnv(render_env, translator=translator, env_id=env_id)
                render_result = render_optimal_policy(
                    eval_result,
                    env=render_env,
                    max_steps=max_steps,
                    seed=0,
                    fallback=_greedy_missing_state,
                    translate=(translator if use_lang_state else True),
                )
            finally:
                render_env.close()
            text_frames = [f.ansi_text for f in render_result.frames if f.ansi_text]
            if text_frames:
                _dash.finish(agent_id, policy_frames=text_frames)
                return
            if rgb_error:
                _dash.finish(
                    agent_id,
                    policy_text=(
                        f"Dashboard render error: rgb_array failed ({rgb_error}) and ansi produced no frames."
                    ),
                )
            else:
                _dash.finish(agent_id, policy_text="Dashboard render error: no rgb_array or ansi frames were produced.")
            return
        except Exception as exc:
            if rgb_error:
                _dash.finish(
                    agent_id,
                    policy_text=(
                        f"Dashboard render error: rgb_array failed ({rgb_error}); ansi failed ({exc})."
                    ),
                )
            else:
                _dash.finish(agent_id, policy_text=f"Dashboard render error (ansi): {exc}")
            return

    except Exception:
        try:
            _dash.finish(agent_id, policy_text="Dashboard render error: unexpected failure during final policy replay.")
        except Exception:
            pass


# Human-readable descriptions shown when the user asks what agents are available.
_AGENT_DESCRIPTIONS: dict[str, str] = {
    "tabular_q": (
        "Tabular Q-learning – lookup-table Q-learning with ε-greedy exploration. "
        "Best for small discrete observation spaces (e.g. Sailing-v0 text strings). "
        "Fast to train, exact, but does not generalise to unseen states."
    ),
    "dqn": (
        "Deep Q-Network (DQN) – two-hidden-layer neural net with experience replay "
        "and a target network.  Works on any flat-vector or text observation; "
        "observations are encoded to a numeric vector automatically.  "
        "Good balance of speed and expressiveness."
    ),
    "ppo": (
        "Proximal Policy Optimisation (PPO) – actor-critic policy-gradient method "
        "with GAE advantage estimation and clipped surrogate objective.  "
        "Robust and sample-efficient; suitable for longer training runs."
    ),
}


@mcp.tool()
def rl_list_agents() -> str:
    """
    List the available RL agent types that can be trained with rl_train_agent().

    Returns a description of each agent's algorithm and when to use it.
    """
    lines = ["Available RL agents:\n"]
    for name, desc in _AGENT_DESCRIPTIONS.items():
        lines.append(f"  • {name}\n      {desc}\n")
    lines.append(
        "Use rl_train_agent(agent_type=..., env_id=...) to train an agent.\n"
        "After training, use rl_run_agent_episode(agent_id=...) to evaluate it."
    )
    return "\n".join(lines)


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

    lines = [f"Saved trained agents — {len(all_agents)} total\n"]
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
def rl_train_agent(
    agent_type: str,
    env_id: str,
    n_episodes: int = 300,
    max_steps: int = 200,
    seed: Optional[int] = None,
    match_id: str = "",
    sub_goal_bonus: float = 0.0,
    sub_goal_threshold: float = 0.5,
    use_language_state: bool = False,
    # Tabular Q hyper-parameters
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    # DQN / PPO shared
    hidden_size: int = 64,
    lr: float = 1e-3,
    # DQN specific
    buffer_size: int = 10000,
    batch_size: int = 64,
    target_update_freq: int = 100,
    # PPO specific
    lr_critic: float = 1e-3,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    n_steps: int = 256,
    ppo_epochs: int = 4,
    mini_batch_size: int = 64,
) -> str:
    """
    Train an RL agent on a registered environment.

    Call rl_list_agents() first to see which agent types are available and
    when to use each one.

    Parameters
    ----------
    agent_type:
        One of "tabular_q", "dqn", or "ppo".
    env_id:
        A registered RLIP environment ID, e.g. "Sailing-v0".
    n_episodes:
        Number of training episodes.
    max_steps:
        Maximum steps per episode.
    seed:
        Optional integer seed for reproducibility.
    match_id:
        Optional match_id returned by rl_match_instruction().  When provided,
        the agent is trained with sub-goal reward shaping: a bonus reward is
        added at every step where the current observation's language description
        has cosine similarity ≥ sub_goal_threshold to the matched sub-goal.
        This combines instruction-following and RL agent training into a single
        call.
    sub_goal_bonus:
        Bonus reward magnitude added when the sub-goal is reached (only used
        when match_id is provided).  Set to 0.0 (default) to auto-scale:
        ``max_reward / (100 × n_sub_goals)`` where *max_reward* is inferred
        from the environment's reward range and *n_sub_goals* is the number
        of matched states.  Pass an explicit positive value to override.
    sub_goal_threshold:
        Cosine similarity threshold to trigger the sub-goal bonus (0–1).
    use_language_state:
        When True, the agent is trained on natural-language descriptions of
        observations instead of the raw numeric/array observations.  The
        environment's registered language translator converts each observation
        to a string before it reaches the agent.  This is ideal for
        tabular_q (which uses strings as Q-table keys directly) and works
        with dqn/ppo via their hash-based encoding fallback.  Requires a
        translator to be registered for *env_id* (see
        rl_set_translator_code).  The language-state flag is stored with the
        agent so rl_run_agent_episode automatically uses the same mode.
    alpha:
        (tabular_q) Q-learning rate.
    gamma:
        Discount factor — shared by all agents.
    epsilon:
        (tabular_q, dqn) Initial exploration rate.
    epsilon_min:
        (tabular_q, dqn) Minimum exploration rate after decay.
    epsilon_decay:
        (tabular_q, dqn) Per-episode multiplicative ε decay.
    hidden_size:
        (dqn, ppo) Hidden-layer width for neural network(s).
    lr:
        (dqn, ppo-actor) Learning rate.
    buffer_size:
        (dqn) Replay buffer capacity.
    batch_size:
        (dqn) Mini-batch size for experience replay.
    target_update_freq:
        (dqn) Steps between hard target-network copies.
    lr_critic:
        (ppo) Critic learning rate.
    lam:
        (ppo) GAE λ parameter.
    clip_eps:
        (ppo) PPO clipping coefficient ε.
    n_steps:
        (ppo) Rollout steps collected per PPO iteration.
    ppo_epochs:
        (ppo) Gradient-update passes per PPO iteration.
    mini_batch_size:
        (ppo) Mini-batch size within each PPO epoch.

    Returns
    -------
    A training summary and an agent_id for use with rl_run_agent_episode().
    """
    from ..rl_agents import TabularQAgent, DQNAgent, PPOAgent  # noqa: PLC0415
    from ..environments.registry import registry as _env_registry  # noqa: PLC0415

    agent_type = agent_type.lower().strip()
    resolved_bonus: Optional[float] = None  # set when sub-goal shaping is active
    if agent_type not in _AGENT_DESCRIPTIONS:
        return (
            f"Unknown agent type '{agent_type}'.  "
            f"Valid choices: {', '.join(_AGENT_DESCRIPTIONS)}.\n"
            "Call rl_list_agents() for details."
        )

    try:
        factory = _env_registry.get(env_id)
        env = factory.create()
    except KeyError:
        return (
            f"Environment '{env_id}' is not registered.  "
            "Call rl_list_environments() to see available environments."
        )

    # ── Optional sub-goal shaping via a cached match_id ───────────────────────
    shaping_summary = ""
    if match_id:
        entry = _instruction_protocols.get(match_id)
        if entry is None:
            return (
                f"match_id '{match_id}' not found.  "
                "Call rl_match_instruction() first to generate a valid match_id."
            )
        protocol = entry["protocol"]
        is_sequential = bool(entry.get("is_sequential"))
        n_sub_goals = 0
        # None tells _ShapedEnv to auto-scale; explicit >0 overrides.
        effective_bonus: Optional[float] = None if sub_goal_bonus == 0.0 else sub_goal_bonus
        # Resolve encoder factory from the match entry (defaults to tfidf).
        _encoder_name = entry.get("encoder_name", "tfidf")
        from ..instruction_matching import get_encoder as _get_encoder  # noqa: PLC0415
        def _encoder_factory(_enc_name=_encoder_name):
            return _get_encoder(_enc_name)
        if is_sequential:
            stage_languages: list[list[str]] = []
            for m in getattr(protocol, "matches", []):
                langs = [lg for lg, _obs, _sc in getattr(m, "matched_states", [])]
                if not langs and getattr(m, "matched_language", None):
                    langs = [m.matched_language]
                if langs:
                    stage_languages.append(langs)
            if not stage_languages:
                return (
                    f"match_id '{match_id}' does not have valid sequential stages. "
                    "Re-run rl_match_instruction() or rl_match_sequential_instructions()."
                )
            n_sub_goals = sum(len(s) for s in stage_languages)
            shaped_env = _SequentialShapedEnv(
                env,
                stage_languages=stage_languages,
                bonus=effective_bonus,
                threshold=sub_goal_threshold,
                translator=getattr(protocol, "translate", True),
                env_id=env_id,
                encoder_factory=_encoder_factory,
            )
            shaped_env._ensure_encoders()
            resolved_bonus = shaped_env._bonus
            env = shaped_env
            shaping_summary = (
                f"\n  Sub-goal shaping: ON (sequential)  (match_id={match_id})\n"
                f"  Stages:           {len(stage_languages)}\n"
                f"  Sub-goals total:  {n_sub_goals} state(s)\n"
                f"  Bonus (auto-scaled): {resolved_bonus:.6g} / threshold={sub_goal_threshold}"
            )
        else:
            n_sub_goals = len(protocol._all_sub_goal_languages)
            # Wrap the environment so that step() injects the similarity bonus.
            shaped_env = _ShapedEnv(
                env,
                sub_goal_language=protocol.sub_goal_language,
                sub_goal_languages=[
                    lg for lg in protocol._all_sub_goal_languages
                    if lg != protocol.sub_goal_language
                ],
                bonus=effective_bonus,
                threshold=sub_goal_threshold,
                translator=protocol.translate,
                env_id=env_id,
                encoder_factory=_encoder_factory,
            )
            # Trigger encoder/bonus resolution now so we can report the value.
            shaped_env._ensure_encoder()
            resolved_bonus = shaped_env._bonus
            env = shaped_env
            shaping_summary = (
                f"\n  Sub-goal shaping: ON  (match_id={match_id})\n"
                f"  Sub-goals:        {n_sub_goals} state(s)\n"
                f"  Primary:          {protocol.sub_goal_language!r}\n"
                f"  Bonus (auto-scaled): {resolved_bonus:.6g} / threshold={sub_goal_threshold}"
            )

    # ── Optional language-state wrapping ─────────────────────────────────────
    lang_state_summary = ""
    if use_language_state:
        from ..language_translation import get_translator  # noqa: PLC0415
        translator = _custom_translators.get(env_id) or get_translator(env_id)
        if translator is None:
            return (
                f"use_language_state=True requires a language translator for "
                f"'{env_id}', but none is registered.\n"
                "Register one with rl_set_translator_code() first."
            )
        env = _LangStateEnv(env, translator=translator, env_id=env_id)
        lang_state_summary = f"\n  Language state:   ON  (translator={type(translator).__name__})"

    # Build the requested agent
    if agent_type == "tabular_q":
        agent = TabularQAgent(
            n_actions=2,          # auto-detected inside train()
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            seed=seed,
        )
    elif agent_type == "dqn":
        agent = DQNAgent(
            hidden_size=hidden_size,
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            seed=seed,
        )
    else:  # ppo
        agent = PPOAgent(
            hidden_size=hidden_size,
            lr_actor=lr,
            lr_critic=lr_critic,
            gamma=gamma,
            lam=lam,
            clip_eps=clip_eps,
            n_steps=n_steps,
            ppo_epochs=ppo_epochs,
            mini_batch_size=mini_batch_size,
            seed=seed,
        )

    agent_id = uuid.uuid4().hex[:12]

    # Register with the live dashboard if it is running
    if _dash_running():
        dash_instructions: list[str] = []
        if match_id and match_id in _instruction_protocols:
            _p = _instruction_protocols[match_id]["protocol"]
            seq_instrs = getattr(_p, "instructions", None)
            if isinstance(seq_instrs, list) and seq_instrs:
                dash_instructions.extend(str(s) for s in seq_instrs[:5])
            else:
                instr = getattr(_p, "instruction", "")
                if instr:
                    dash_instructions.append(str(instr))
        _dash.register(
            agent_id,
            agent_type=agent_type,
            env_id=env_id,
            n_episodes=n_episodes,
            use_language_state=use_language_state,
            uses_instructions=bool(match_id),
            instructions=dash_instructions,
        )

    progress_env = _ProgressEnv(
        env,
        n_episodes=n_episodes,
        agent_type=agent_type,
        env_id=env_id,
        dashboard_agent_id=agent_id if _dash_running() else "",
        agent_ref=agent,
    )

    job_id = uuid.uuid4().hex[:12]
    _training_jobs[job_id] = {
        "status": "running",
        "agent_id": agent_id,
        "env_id": env_id,
        "n_episodes": n_episodes,
        "progress_env": progress_env,
    }

    def _job() -> None:
        try:
            result = agent.train(
                progress_env,
                n_episodes=n_episodes,
                max_steps=max_steps,
                seed=seed,
            )
        except Exception as exc:
            _training_jobs[job_id]["status"] = "failed"
            _training_jobs[job_id]["result"] = f"Training failed: {exc}"
            try:
                progress_env.close()
            except Exception:
                pass
            return

        try:
            progress_env.close()
        except Exception:
            pass

        instructions_used = _collect_training_instructions(match_id)

        _trained_agents[agent_id] = {
            "agent":                agent,
            "env_id":               env_id,
            "agent_type":           agent_type,
            "best_episode_history": result.best_episode_history,
            "use_language_state":   use_language_state,
            "train_result":         result,
            "instructions_used":    instructions_used,
            "training_config": {
                "n_episodes": n_episodes,
                "max_steps": max_steps,
                "seed": seed,
                "match_id": match_id or None,
                "sub_goal_bonus": resolved_bonus,
                "sub_goal_threshold": sub_goal_threshold if match_id else None,
                "use_language_state": use_language_state,
                "alpha": alpha if agent_type == "tabular_q" else None,
                "gamma": gamma,
                "epsilon": epsilon if agent_type in {"tabular_q", "dqn"} else None,
                "epsilon_min": epsilon_min if agent_type in {"tabular_q", "dqn"} else None,
                "epsilon_decay": epsilon_decay if agent_type in {"tabular_q", "dqn"} else None,
                "hidden_size": hidden_size if agent_type in {"dqn", "ppo"} else None,
                "lr": lr if agent_type in {"dqn", "ppo"} else None,
                "buffer_size": buffer_size if agent_type == "dqn" else None,
                "batch_size": batch_size if agent_type == "dqn" else None,
                "target_update_freq": target_update_freq if agent_type == "dqn" else None,
                "lr_critic": lr_critic if agent_type == "ppo" else None,
                "lam": lam if agent_type == "ppo" else None,
                "clip_eps": clip_eps if agent_type == "ppo" else None,
                "n_steps": n_steps if agent_type == "ppo" else None,
                "ppo_epochs": ppo_epochs if agent_type == "ppo" else None,
                "mini_batch_size": mini_batch_size if agent_type == "ppo" else None,
            },
            "match_id":             match_id or None,
            "sub_goal_language":    (
                getattr(_instruction_protocols[match_id]["protocol"], "sub_goal_language", None)
                if match_id and match_id in _instruction_protocols else None
            ),
            "sub_goal_bonus":       resolved_bonus,
            "sub_goal_threshold":   sub_goal_threshold if match_id else None,
            "instruction":          (
                (
                    " -> ".join(getattr(_instruction_protocols[match_id]["protocol"], "instructions", [])[:5])
                    if getattr(_instruction_protocols[match_id]["protocol"], "instructions", None)
                    else getattr(_instruction_protocols[match_id]["protocol"], "instruction", None)
                )
                if match_id and match_id in _instruction_protocols else None
            ),
            "n_subgoals":           (
                (
                    sum(len(getattr(m, "matched_states", []) or [getattr(m, "matched_language", "")])
                        for m in getattr(_instruction_protocols[match_id]["protocol"], "matches", []))
                    if getattr(_instruction_protocols[match_id]["protocol"], "matches", None)
                    else len(getattr(_instruction_protocols[match_id]["protocol"], "_all_sub_goal_languages", []))
                )
                if match_id and match_id in _instruction_protocols else None
            ),
            "best_match_observation": (
                (_instruction_protocols.get(match_id, {}).get("match_summary") or {}).get("best_match_observation")
                if match_id else None
            ),
            "best_match_similarity": (
                (_instruction_protocols.get(match_id, {}).get("match_summary") or {}).get("best_match_similarity")
                if match_id else None
            ),
            "original_instruction": (
                _instruction_protocols.get(match_id, {}).get("original_instruction")
                if match_id else None
            ),
        }

        if _dash_running():
            _render_policy_for_dashboard(agent_id, env_id, result.best_episode_history, max_steps)

        artifact_path: str | None = None
        artifact_warning = ""
        archive_path, archive_error = _package_trained_agent(agent_id, _trained_agents[agent_id])
        if archive_path is not None:
            artifact_path = str(archive_path)
            _trained_agents[agent_id]["artifact_archive"] = artifact_path
        elif archive_error:
            artifact_warning = f"\n  Artifact package: FAILED ({archive_error})"

        # ── Clean 100-episode evaluation (fixed weights, no instruction rewards) ──
        _eval_mean: float | None = None
        _eval_std: float | None = None
        _eval_rewards: list[float] = []
        try:
            from ..environments.registry import registry as _eval_env_registry  # noqa: PLC0415
            from ..language_translation import get_translator as _get_translator  # noqa: PLC0415

            _eval_factory = _eval_env_registry.get(env_id)
            _eval_agent = _trained_agents[agent_id]["agent"]
            _eval_translator = (
                _custom_translators.get(env_id) or _get_translator(env_id)
                if use_language_state else None
            )
            _eval_mean, _eval_std, _eval_rewards = _run_clean_evaluation(
                agent=_eval_agent,
                env_factory=_eval_factory,
                n_episodes=100,
                max_steps=max_steps,
                use_language_state=use_language_state,
                translator=_eval_translator,
                env_id=env_id,
                seed=seed if seed is not None else 0,
            )
            _trained_agents[agent_id]["eval_mean"] = _eval_mean
            _trained_agents[agent_id]["eval_std"] = _eval_std
            _trained_agents[agent_id]["eval_rewards"] = _eval_rewards
        except Exception:
            pass

        # Update plan database if this agent was trained with an instruction
        if match_id and match_id in _instruction_protocols and _eval_mean is not None:
            try:
                from ..instruction_following import get_plan_database  # noqa: PLC0415
                from ._state import _env_plan_db_path  # noqa: PLC0415

                _proto_entry = _instruction_protocols[match_id]
                _original_instruction = (
                    _proto_entry.get("original_instruction")
                    or _proto_entry.get("match_summary", {}).get("instruction")
                    or ""
                )
                if _original_instruction:
                    _plan_db = get_plan_database(env_id, plan_path=str(_env_plan_db_path(env_id)))
                    _plan_db.update_eval_reward(
                        instruction=_original_instruction,
                        match_id=match_id,
                        eval_reward=_eval_mean,
                        training_reward=result.best_reward,
                        agent_id=agent_id,
                        agent_type=agent_type,
                        n_episodes=n_episodes,
                    )
            except Exception:
                pass

        try:
            _match_summary = (_instruction_protocols.get(match_id, {}).get("match_summary") or {}) if match_id else {}
            _reg_entry = {
                "agent_id":             agent_id,
                "agent_type":           agent_type,
                "env_id":               env_id,
                "created_at":           datetime.now().isoformat(),
                "instruction":          _trained_agents[agent_id].get("original_instruction")
                                        or _trained_agents[agent_id].get("instruction"),
                "sub_steps":            (_instruction_protocols.get(match_id, {}).get("instructions") or [])
                                        if match_id else [],
                "eval_reward":          _eval_mean,
                "eval_std":             _eval_std,
                "eval_n_episodes":      len(_eval_rewards) if _eval_rewards else None,
                "training_best_reward": result.best_reward,
                "match_similarity":     _trained_agents[agent_id].get("best_match_similarity"),
                "matched_language":     _match_summary.get("best_match_language"),
                "n_episodes":           n_episodes,
                "artifact_path":        artifact_path,
            }
            _reg_path = _env_agents_registry_path(env_id)
            _reg_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                _reg_data = json.loads(_reg_path.read_text(encoding="utf-8")) if _reg_path.exists() else {}
            except Exception:
                _reg_data = {}
            _reg_data[agent_id] = _reg_entry
            _reg_path.write_text(json.dumps(_reg_data, indent=2), encoding="utf-8")
        except Exception:
            pass

        eval_summary = ""
        if _eval_mean is not None and _eval_std is not None:
            eval_summary = (
                f"\n  Clean evaluation (100 eps, fixed weights, no instruction rewards):\n"
                f"    mean reward = {_eval_mean:.4f}  ±  {_eval_std:.4f}  (std)\n"
            )

        summary = (
            f"Training complete — {agent_type} on {env_id}{shaping_summary}{lang_state_summary}\n\n"
            f"  {result}\n\n"
            f"  Mean reward (last 10 %): {result.last_n_mean:.4f}\n"
            f"{eval_summary}"
            f"  Agent ID: {agent_id}\n\n"
        )
        if artifact_path:
            summary += f"  Artifact package: {artifact_path}\n\n"
        if artifact_warning:
            summary += f"{artifact_warning}\n"
        summary += f"Use rl_run_agent_episode(agent_id='{agent_id}') to evaluate the agent."

        _training_jobs[job_id]["result"] = summary
        _training_jobs[job_id]["status"] = "done"

    threading.Thread(target=_job, daemon=True).start()

    return (
        f"Training started — {agent_type} on {env_id}{shaping_summary}{lang_state_summary}\n\n"
        f"  Agent ID:  {agent_id}\n"
        f"  Job ID:    {job_id}\n"
        f"  Episodes:  {n_episodes}  max_steps={max_steps}\n\n"
        f"Training runs in the background with no time limits.\n"
        f"Call rl_get_training_result(job_id='{job_id}') to check progress and get the result."
    )


@mcp.tool()
def rl_get_training_result(job_id: str) -> str:
    """
    Check the status of a background training job started by rl_train_agent
    or rl_train_and_derive_instructions.

    Training runs in a background thread with no time limits so that long
    training runs are never interrupted by tool timeouts.  Call this tool
    after starting training to check progress, then again when it is done to
    retrieve the full result.

    Parameters
    ----------
    job_id:
        The job_id returned by rl_train_agent() or
        rl_train_and_derive_instructions().

    Returns
    -------
    If still running: episode progress and the agent_id.
    If complete: the full training result summary (same as the old direct
    return value).
    If failed: the error message.
    """
    job = _training_jobs.get(job_id)
    if not job:
        active = list(_training_jobs.keys())
        return (
            f"No training job found with ID '{job_id}'.\n"
            f"Active job IDs: {active if active else 'none'}"
        )
    if job["status"] == "running":
        progress_env = job.get("progress_env")
        done = getattr(progress_env, "_completed_episodes", 0) if progress_env else 0
        total = job.get("n_episodes", "?")
        pct = f"{done / total * 100:.0f}%" if isinstance(total, int) and total > 0 else "?"
        return (
            f"Training in progress: {done}/{total} episodes ({pct})\n"
            f"  Agent ID: {job.get('agent_id', '?')}\n"
            f"  Env:      {job.get('env_id', '?')}\n\n"
            f"Call rl_get_training_result(job_id='{job_id}') again to check."
        )
    return job.get("result", "No result available.")


@mcp.tool()
def rl_load_agent(artifact_path: str = "", agent_id: str = "") -> str:
    """
    Load a previously saved RL agent from disk into this session.

    Accepts either:
    • An artifact ZIP path (as returned by rl_train_agent or shown in
      rl_list_trained_agents) — the path to a ``<agent_type>_<id>_<ts>.zip``
      file or the unzipped package directory containing weights/ and
      metadata.json.
    • An agent_id from the persistent agents registry — the registry entry
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
    import zipfile
    import tempfile
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

    # ── Register in session cache ─────────────────────────────────────────────
    _trained_agents[loaded_agent_id] = {
        "agent":                agent,
        "env_id":               env_id,
        "agent_type":           agent_type,
        "use_language_state":   use_lang_state,
        "train_result":         None,
        "best_episode_history": [],
        "training_config":      training_config,
        "match_id":             metadata.get("match_id"),
        "instruction":          metadata.get("instruction"),
        "original_instruction": instruction,
        "sub_goal_language":    metadata.get("sub_goal_language"),
        "sub_goal_bonus":       metadata.get("sub_goal_bonus"),
        "sub_goal_threshold":   metadata.get("sub_goal_threshold"),
        "artifact_archive":     str(resolved_path) if resolved_path.suffix.lower() == ".zip" else None,
        "loaded_from":          str(resolved_path),
    }

    instr_line = f"\n  Instruction:  {instruction!r}" if instruction else ""
    lang_line  = "\n  Language obs: ON" if use_lang_state else ""

    return (
        f"Agent loaded successfully.\n\n"
        f"  Agent ID:    {loaded_agent_id}\n"
        f"  Type:        {agent_type}\n"
        f"  Environment: {env_id}"
        + instr_line
        + lang_line
        + f"\n\nThe agent is ready to use:\n"
        f"  rl_run_agent_episode(agent_id='{loaded_agent_id}')\n"
        f"  rl_render_policy(env_id='{env_id}', agent_id='{loaded_agent_id}')\n"
        f"  rl_create_training_report(agent_id='{loaded_agent_id}')"
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
        If True, use stochastic (sampled) action selection — meaningful for
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
        f"Evaluation episode — {agent_type} on {env_id}\n"
        f"  Agent ID:     {agent_id}\n"
        f"  Action mode:  {mode_str}\n"
        f"  Obs mode:     {lang_mode}\n"
        f"  Steps:        {step_count}\n"
        f"  Total reward: {total_reward:.4f}\n"
        f"  End reason:   {end_reason}\n\n"
        "Trajectory:\n" + "\n".join(trajectory)
    )


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
        ``<cwd>/rlip_results/<env>/reports/<env>_<agent_id>_report.png``.
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
        out_path = output_path or str(env_reports_dir / f"{safe_id}_comparison_report.png")
    else:
        out_path = output_path or str(
            env_reports_dir / f"{safe_id}_{primary_agent_type}_{agent_id}_report.png"
        )

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
            }
        )

    try:
        fig = create_training_report(
            comparison_runs=comparison_runs,
            output_path=out_path,
            rolling_window=rolling_window or None,
            n_breakpoints=breakpoint_count,
        )
    except Exception as exc:
        return f"Report generation failed: {exc}"

    # Return base64-encoded PNG so Claude can display inline
    try:
        import io as _io  # noqa: PLC0415
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return (
            f"Training report generated for {len(entries)} agent(s) on {env_id}.\n"
            f"Agent IDs: {', '.join(requested_ids)}\n"
            f"Saved to: {out_path}\n\n"
            f"data:image/png;base64,{b64}"
        )
    except Exception as exc:
        return f"Report saved to {out_path} but base64 encoding failed: {exc}"


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
    - Load the trained agent's weights (fixed — no further learning).
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
        return "Evaluation produced no completed episodes — check environment and agent compatibility."

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
        f"Clean evaluation — {agent_type} on {env_id}\n"
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
