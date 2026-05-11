"""
RL agent training and evaluation MCP tools for the RLIP plugin.

Tools: rl_list_agents, rl_train_agent, rl_run_agent_episode,
       rl_create_training_report.
"""

from __future__ import annotations

import base64
from typing import Any, Optional

from ._env_wrappers import _LangStateEnv, _ShapedEnv
from ._state import (
    _RENDERS_DIR,
    _custom_translators,
    _in_process,
    _instruction_protocols,
    _trained_agents,
    log,
    mcp,
)

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
def rl_train_agent(
    agent_type: str,
    env_id: str,
    n_episodes: int = 300,
    max_steps: int = 200,
    seed: Optional[int] = None,
    match_id: str = "",
    sub_goal_bonus: float = 1.0,
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
        when match_id is provided).
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
    import uuid  # noqa: PLC0415
    from ..rl_agents import TabularQAgent, DQNAgent, PPOAgent  # noqa: PLC0415
    from ..environments.registry import registry as _env_registry  # noqa: PLC0415

    agent_type = agent_type.lower().strip()
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
        # Wrap the environment so that step() injects the similarity bonus.
        env = _ShapedEnv(
            env,
            sub_goal_language=protocol.sub_goal_language,
            sub_goal_languages=[
                lg for lg in protocol._all_sub_goal_languages
                if lg != protocol.sub_goal_language
            ],
            bonus=sub_goal_bonus,
            threshold=sub_goal_threshold,
            translator=protocol.translate,
            env_id=env_id,
        )
        shaping_summary = (
            f"\n  Sub-goal shaping: ON  (match_id={match_id})\n"
            f"  Sub-goals:        {len(protocol._all_sub_goal_languages)} state(s)\n"
            f"  Primary:          {protocol.sub_goal_language!r}\n"
            f"  Bonus / threshold:{sub_goal_bonus} / {sub_goal_threshold}"
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

    result = agent.train(env, n_episodes=n_episodes, max_steps=max_steps, seed=seed)

    agent_id = uuid.uuid4().hex[:12]
    _trained_agents[agent_id] = {
        "agent":                agent,
        "env_id":               env_id,
        "agent_type":           agent_type,
        "best_episode_history": result.best_episode_history,
        "use_language_state":   use_language_state,
        "train_result":         result,
        # Sub-goal metadata (populated when match_id was provided)
        "match_id":             match_id or None,
        "sub_goal_language":    (
            _instruction_protocols[match_id]["protocol"].sub_goal_language
            if match_id and match_id in _instruction_protocols else None
        ),
        "sub_goal_bonus":       sub_goal_bonus if match_id else None,
        "sub_goal_threshold":   sub_goal_threshold if match_id else None,
        "instruction":          (
            _instruction_protocols[match_id]["protocol"].instruction
            if match_id and match_id in _instruction_protocols else None
        ),
        "n_subgoals":           (
            len(_instruction_protocols[match_id]["protocol"]._all_sub_goal_languages)
            if match_id and match_id in _instruction_protocols else None
        ),
    }

    return (
        f"Training complete — {agent_type} on {env_id}{shaping_summary}{lang_state_summary}\n\n"
        f"  {result}\n\n"
        f"  Mean reward (last 10 %): {result.last_n_mean:.4f}\n"
        f"  Agent ID: {agent_id}\n\n"
        f"Use rl_run_agent_episode(agent_id='{agent_id}') to evaluate the agent."
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
    output_path: str = "",
    rolling_window: int = 0,
    n_sample_frames: int = 6,
    render_for_frames: bool = True,
    n_render_episodes: int = 20,
    fps: float = 6.0,
) -> str:
    """
    Generate a multi-panel training report for a previously trained agent and
    save it as a PNG image.

    The report contains four sections:

    1. **Reward curve** — per-episode reward and rolling average over the full
       training run, with the best episode marked.
    2. **Sample frames** — evenly-spaced RGB frames from the best episode
       replayed through the learnt policy (only if the environment supports
       rgb_array rendering; requires render_for_frames=True).
    3. **Metadata table** — agent type, hyper-parameters, training statistics,
       and sub-goal shaping settings.
    4. **Sub-goal similarity panel** — if the agent was trained with a
       match_id (via rl_match_instruction), shows per-step cosine similarity
       to the sub-goal during the best training episode, with markers at each
       step the reward bonus was triggered.  When no sub-goal was used, a
       placeholder is shown.

    Parameters
    ----------
    agent_id:
        The agent_id returned by rl_train_agent().
    output_path:
        Path to write the PNG report.  Defaults to
        ``~/.rlip/renders/<env>_<agent_id>_report.png``.
    rolling_window:
        Number of episodes for the rolling reward average (0 = auto: 5 %
        of total episodes, minimum 10).
    n_sample_frames:
        Number of evenly-spaced frames to show in the frames strip (max 8).
    render_for_frames:
        Set False to skip the rgb_array rendering step (faster, but no
        frame strip in the report).
    n_render_episodes:
        Number of episodes to run before selecting the best one to render.
    fps:
        Frame rate used when the GIF companion file is also desired.

    Returns
    -------
    The local path of the saved PNG report and its base64-encoded content
    so Claude can display it inline.
    """
    from ..analysis import create_training_report  # noqa: PLC0415
    from ..environments.registry import registry as _env_registry  # noqa: PLC0415
    from ..instruction_matching import TextEncoder  # noqa: PLC0415
    from ..language_translation import get_translator  # noqa: PLC0415

    entry = _trained_agents.get(agent_id)
    if entry is None:
        return (
            f"Agent ID '{agent_id}' not found.  "
            "Run rl_train_agent() first to train an agent."
        )

    env_id     = entry["env_id"]
    agent_type = entry["agent_type"]
    train_result = entry.get("train_result")

    if train_result is None:
        return (
            "No training result stored for this agent.  "
            "This agent was trained with an older version of RLIP; re-train to "
            "enable reporting."
        )

    # ── Determine output path ─────────────────────────────────────────────────
    _RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id   = env_id.replace("/", "_").replace("-", "_").replace(" ", "_")
    out_path  = output_path or str(
        _RENDERS_DIR / f"{safe_id}_{agent_type}_{agent_id}_report.png"
    )

    # ── Render best episode to get sample frames ──────────────────────────────
    render_result = None
    if render_for_frames and _in_process:
        try:
            from ..interaction_protocols import (  # noqa: PLC0415
                GreedyEpisodeProtocol,
                MultiEpisodeProtocol,
                RandomEpisodeProtocol,
            )
            from ..policy_rendering import render_optimal_policy  # noqa: PLC0415

            factory   = _env_registry.get(env_id)
            train_env = factory.create(render_mode=None)
            try:
                stored_agent = entry["agent"]
                best_history = entry.get("best_episode_history", [])
                if best_history:
                    def _greedy_fn(obs: Any) -> Any:  # noqa: E731
                        act_fn = getattr(stored_agent, "act_greedy", None) or stored_agent.act
                        return act_fn(obs)
                    base = GreedyEpisodeProtocol(
                        policy_fn=_greedy_fn, max_steps=200, record_history=True
                    )
                else:
                    base = RandomEpisodeProtocol(max_steps=200, record_history=True)
                protocol = MultiEpisodeProtocol(base, n_episodes=n_render_episodes)
                collection_result = protocol(train_env)
            finally:
                train_env.close()

            render_result = render_optimal_policy(
                collection_result,
                env_factory=factory,
                render_mode="rgb_array",
                max_steps=200,
            )
        except Exception as _render_exc:
            log.warning("Frame rendering failed (non-fatal): %s", _render_exc)

    # ── Compute per-step sub-goal similarity from best training episode ────────
    subgoal_steps: list[tuple[int, float, bool]] = []
    subgoal_info:  dict[str, Any] | None = None

    match_id_stored = entry.get("match_id")
    sub_goal_lang   = entry.get("sub_goal_language")
    if match_id_stored and sub_goal_lang:
        instruction = entry.get("instruction", "")
        threshold   = float(entry.get("sub_goal_threshold") or 0.5)
        bonus       = entry.get("sub_goal_bonus")
        n_subgoals  = entry.get("n_subgoals")

        subgoal_info = {
            "instruction":       instruction,
            "sub_goal_language": sub_goal_lang,
            "threshold":         threshold,
            "bonus":             bonus,
            "match_id":          match_id_stored,
            "n_subgoals":        n_subgoals,
        }

        # Re-compute similarity for each step of the best training episode.
        use_lang_state = entry.get("use_language_state", False)
        translator = _custom_translators.get(env_id) or get_translator(env_id)
        best_history = entry.get("best_episode_history", [])

        if best_history and (translator or use_lang_state):
            try:
                enc = TextEncoder()
                enc.fit([sub_goal_lang])
                sg_vec = enc.encode(sub_goal_lang)

                for step_n, (obs, _action) in enumerate(best_history, 1):
                    # If language-state was used, obs is already a string
                    if use_lang_state and isinstance(obs, str):
                        lang = obs
                    elif translator:
                        lang = translator.translate(obs) or ""
                    else:
                        lang = ""

                    if not lang:
                        continue

                    obs_vec = enc.encode(lang)
                    sim = float(enc.cosine_similarity(obs_vec, sg_vec))
                    reached = sim >= threshold
                    subgoal_steps.append((step_n, sim, reached))
            except Exception as _sg_exc:
                log.warning("Sub-goal similarity computation failed: %s", _sg_exc)

    # ── Extra metadata shown in the table ─────────────────────────────────────
    extra_meta: dict[str, Any] = {"Agent ID": agent_id}
    if entry.get("use_language_state"):
        extra_meta["Language state"] = "Yes"
    if match_id_stored:
        extra_meta["Match ID"] = match_id_stored
        extra_meta["Sub-goal shaping"] = "Yes"

    # ── Generate report ───────────────────────────────────────────────────────
    try:
        fig = create_training_report(
            train_result=train_result,
            render_result=render_result,
            metadata=extra_meta,
            subgoal_steps=subgoal_steps or None,
            subgoal_info=subgoal_info,
            output_path=out_path,
            rolling_window=rolling_window or None,
            n_sample_frames=min(n_sample_frames, 8),
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
            f"Training report generated for agent '{agent_id}' ({agent_type} on {env_id}).\n"
            f"Saved to: {out_path}\n\n"
            f"data:image/png;base64,{b64}"
        )
    except Exception as exc:
        return f"Report saved to {out_path} but base64 encoding failed: {exc}"
