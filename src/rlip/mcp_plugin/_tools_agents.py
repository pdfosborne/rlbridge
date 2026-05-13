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
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import Context

from ._dashboard import dashboard as _dash, is_running as _dash_running
from ._env_wrappers import _LangStateEnv, _SequentialShapedEnv, _ShapedEnv
from ._state import (
    _CUSTOM_ENV_CACHE_ROOT,
    _env_agents_dir,
    _env_cache_dir,
    _env_renders_dir,
    _custom_translators,
    _instruction_protocols,
    _trained_agents,
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

        custom_env_dir = _CUSTOM_ENV_CACHE_ROOT / env_id
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

        custom_translator = _CUSTOM_ENV_CACHE_ROOT / env_id / "translator.py"
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
async def rl_train_agent(
    ctx: Context,
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

    import uuid as _uuid  # noqa: PLC0415
    agent_id = _uuid.uuid4().hex[:12]

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

    async def _poll_training() -> None:
        while True:
            await asyncio.sleep(1.5)
            await ctx.report_progress(
                progress_env._completed_episodes, n_episodes
            )

    poll_task = asyncio.create_task(_poll_training())
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: agent.train(
                progress_env,
                n_episodes=n_episodes,
                max_steps=max_steps,
                seed=seed,
            ),
        )
    finally:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass
        progress_env.close()

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
        # Sub-goal metadata (populated when match_id was provided)
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
    }

    # Push policy render to dashboard now so the final card always includes
    # either the optimal replay or a concrete render error.
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

    summary = (
        f"Training complete — {agent_type} on {env_id}{shaping_summary}{lang_state_summary}\n\n"
        f"  {result}\n\n"
        f"  Mean reward (last 10 %): {result.last_n_mean:.4f}\n"
        f"  Agent ID: {agent_id}\n\n"
    )
    if artifact_path:
        summary += f"  Artifact package: {artifact_path}\n\n"
    if artifact_warning:
        summary += f"{artifact_warning}\n"
    summary += f"Use rl_run_agent_episode(agent_id='{agent_id}') to evaluate the agent."
    return summary


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
        ``./.rlip/environments/<env>/renders/<env>_<agent_id>_report.png``.
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

    env_render_dir = _env_renders_dir(env_id)
    env_render_dir.mkdir(parents=True, exist_ok=True)
    safe_id = env_id.replace("/", "_").replace("-", "_").replace(" ", "_")
    if len(entries) > 1:
        out_path = output_path or str(env_render_dir / f"{safe_id}_comparison_report.png")
    else:
        out_path = output_path or str(
            env_render_dir / f"{safe_id}_{primary_agent_type}_{agent_id}_report.png"
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
