"""
RL agent utility MCP tools for the rlbridge plugin.

Tools: rl_list_agents

Save/load tools (rl_list_trained_agents, rl_load_agent) and packaging
helpers (_copy_if_exists, _package_trained_agent) live in _tools_agents_io.py.
"""

from __future__ import annotations

from typing import Any, Optional

from ._dashboard import dashboard as _dash
from ._env_wrappers import _LangStateEnv, _SequentialShapedEnv
from ._prompts import AGENT_DESCRIPTIONS as _AGENT_DESCRIPTIONS
from ._state import (
    _env_cache_dir,
    _env_renders_dir,
    _custom_translators,
    _instruction_protocols,
    _trained_agents,
    mcp,
)

# ── Training helper functions ─────────────────────────────────────────────────


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

        # Reconstruct stage_languages for sub_goal highlighting in replay
        match_id = stored.get("match_id")
        sub_goal_threshold = stored.get("sub_goal_threshold") or 0.5
        stage_languages: list[list[str]] | None = None
        if match_id and match_id in _instruction_protocols:
            proto = _instruction_protocols[match_id]["protocol"]
            matches = getattr(proto, "matches", None)
            if matches:
                stage_languages = [
                    [lg for lg, _obs, _sc in m.matched_states] or [m.matched_language]
                    for m in matches
                ]
        if stage_languages is None and stored.get("instructions_used"):
            stage_languages = [[instr] for instr in stored["instructions_used"]]

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
            if stage_languages and translator:
                eval_env = _SequentialShapedEnv(
                    eval_env,
                    stage_languages=stage_languages,
                    bonus=None,
                    threshold=sub_goal_threshold,
                    translator=translator,
                    env_id=env_id,
                )

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
                n_episodes=100,
                base_seed=0,
            )
            eval_result = eval_protocol(eval_env)
        finally:
            eval_env.close()

        if not getattr(eval_result, "episodes", None):
            _dash.finish(agent_id, policy_text="Dashboard render error: evaluation produced no episodes.")
            return

        # Deterministic non-random fallback for replay when the recorded action
        # sequence runs out (e.g. render max_steps > episode length).
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
                if stage_languages and translator:
                    render_env = _SequentialShapedEnv(
                        render_env,
                        stage_languages=stage_languages,
                        bonus=None,
                        threshold=sub_goal_threshold,
                        translator=translator,
                        env_id=env_id,
                )
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
                frame_meta = [
                    {
                        "sub_goal_reached":    f.sub_goal_reached,
                        "sub_goal_similarity": f.sub_goal_similarity,
                        "language_obs":        f.language_obs,
                        "reward":              f.reward,
                        "step":                f.step,
                        "instruction_index":   f.instruction_index,
                    }
                    for f in render_result.frames
                ]
                _dash.finish(agent_id, policy_gif_b64=b64, policy_frame_meta=frame_meta)
                return
        except Exception as exc:
            rgb_error = str(exc)

        # Attempt 2: ANSI text frames (still deterministic, no random fallback)
        try:
            render_env = factory.create(render_mode="ansi")
            try:
                if use_lang_state:
                    render_env = _LangStateEnv(render_env, translator=translator, env_id=env_id)
                if stage_languages and translator:
                    render_env = _SequentialShapedEnv(
                        render_env,
                        stage_languages=stage_languages,
                        bonus=None,
                        threshold=sub_goal_threshold,
                        translator=translator,
                        env_id=env_id,
                    )
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
            frame_data = [(f.ansi_text, f) for f in render_result.frames if f.ansi_text]
            if frame_data:
                text_frames = [t for t, _ in frame_data]
                frame_meta = [
                    {
                        "sub_goal_reached":    f.sub_goal_reached,
                        "sub_goal_similarity": f.sub_goal_similarity,
                        "language_obs":        f.language_obs,
                        "reward":              f.reward,
                        "step":                f.step,
                        "instruction_index":   f.instruction_index,
                    }
                    for _, f in frame_data
                ]
                _dash.finish(agent_id, policy_frames=text_frames, policy_frame_meta=frame_meta)
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


def _build_agent(
    agent_type: str,
    alpha: float, gamma: float,
    epsilon: float, epsilon_min: float, epsilon_decay: float,
    hidden_size: int, lr: float,
    seed: Optional[int],
) -> Any:
    from ..rl_agents import TabularQAgent, DQNAgent, PPOAgent  # noqa: PLC0415
    if agent_type == "tabular_q":
        return TabularQAgent(
            alpha=alpha, gamma=gamma,
            epsilon=epsilon, epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay, seed=seed,
        )
    if agent_type == "dqn":
        return DQNAgent(
            hidden_size=hidden_size, lr=lr, gamma=gamma,
            epsilon=epsilon, epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay, seed=seed,
        )
    # ppo
    from ..rl_agents import PPOAgent  # noqa: PLC0415
    return PPOAgent(hidden_size=hidden_size, lr_actor=lr, gamma=gamma, seed=seed)


@mcp.tool()
def rl_list_agents() -> str:
    """
    List the available RL agent types that can be trained with rl_experiment_process() or rl_train_agent().

    Returns a description of each agent's algorithm and when to use it.
    """
    lines = ["Available RL agents:\n"]
    for name, desc in _AGENT_DESCRIPTIONS.items():
        lines.append(f"  • {name}\n      {desc}\n")
    lines.append(
        "Use rl_experiment_process(agent_type=..., env_id=...) to train an agent (recommended).\n"
        "After training, use rl_run_agent_episode(agent_id=...) to evaluate it."
    )
    return "\n".join(lines)



