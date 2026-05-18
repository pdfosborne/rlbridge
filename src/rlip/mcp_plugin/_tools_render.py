"""
Rendering MCP tools and resources for the RLIP plugin.

Resources: environments_resource, instances_resource.
Tools:     rl_render_policy.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Optional

from ._env_wrappers import _LangStateEnv
from ._dispatch import _dispatch
from ._state import (
    _custom_translators,
    _env_renders_dir,
    _in_process,
    _registry,
    _trained_agents,
    log,
    mcp,
)


# ── MCP Resources ─────────────────────────────────────────────────────────────

@mcp.resource("rlip://environments")
def environments_resource() -> str:
    """All registered RLIP environments as JSON."""
    from ..protocol.constants import Methods
    result = _dispatch(Methods.LIST_ENVIRONMENTS, tags=[], namespace=None)
    return json.dumps(result, indent=2)


@mcp.resource("rlip://instances")
def instances_resource() -> str:
    """All active RLIP environment instances as JSON."""
    from ..protocol.constants import Methods
    result = _dispatch(Methods.LIST_INSTANCES)
    return json.dumps(result, indent=2)


# ── Render tool ───────────────────────────────────────────────────────────────

@mcp.tool()
def rl_render_policy(
    env_id: str,
    n_episodes: int = 30,
    max_steps: int = 200,
    seed: Optional[int] = None,
    fps: float = 6.0,
    agent_id: str = "",
) -> str:
    """
    Render the optimal policy for an environment as an animated GIF.

    NOTE: For the standard single-image render, prefer rl_render_policy_overlay
    instead.  Use this tool only when an animation is explicitly requested.

    Runs several episodes, picks the best one, replays it with rendering
    enabled, and saves the result as a ``.gif`` file.  The GIF is returned
    as an inline data URL so Claude can display it directly, and also saved
    to ``./<cwd>/rlip_results/<env>/renders/`` for local access.

    Parameters
    ----------
    env_id:
        Environment to render (must be visible in rl_list_environments).
    n_episodes:
        Number of episodes to collect before selecting the best policy.
        More episodes → higher chance of finding a good trajectory.
    max_steps:
        Hard cap on each episode length.
    seed:
        Random seed for reproducibility.
    fps:
        Animation speed of the output GIF (frames per second).
    agent_id:
        Optional ID of a trained agent (from rl_train_agent).  When
        provided, the agent's greedy policy is used instead of random
        exploration.

    Returns the GIF as a data URL (``data:image/gif;base64,...``) that
    Claude can display, plus the local file path.
    """
    from datetime import datetime

    from ..interaction_protocols import (
        MultiEpisodeProtocol,
        RandomEpisodeProtocol,
        GreedyEpisodeProtocol,
    )
    from ..policy_rendering import render_optimal_policy

    # ── Resolve environment factory ───────────────────────────────────────────
    if not _registry or env_id not in _registry:
        return f"Unknown environment '{env_id}'. Use rl_list_environments() to see available envs."

    factory = _registry.get(env_id)

    # ── Determine GIF output path ─────────────────────────────────────────────
    env_render_dir = _env_renders_dir(env_id)
    env_render_dir.mkdir(parents=True, exist_ok=True)
    safe_id = env_id.replace("/", "_").replace("-", "_").replace(" ", "_")
    _stored = _trained_agents.get(agent_id) if agent_id else None
    agent_type_label = _stored.get("agent_type", "unknown") if _stored else "random"
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = env_render_dir / f"{safe_id}_{agent_type_label}_{n_episodes}ep_{_ts}.gif"

    # ── Collect episodes ──────────────────────────────────────────────────────
    train_env = factory.create(render_mode=None)
    try:
        if agent_id and agent_id in _trained_agents:
            # Use stored best-training-episode history directly.
            # This avoids re-running the greedy policy, which may cycle for
            # agents like TabularQ whose Q-values didn't fully propagate.
            stored = _trained_agents[agent_id]
            use_lang_state = bool(stored.get("use_language_state", False))
            translator = None
            if use_lang_state:
                from ..language_translation import get_translator  # noqa: PLC0415

                translator = _custom_translators.get(env_id) or get_translator(env_id)
                if translator is None:
                    return (
                        f"Agent '{agent_id}' was trained with use_language_state=True, "
                        f"but no translator is registered for '{env_id}'.\n"
                        "Register one with rl_set_translator_code() first."
                    )

            training_history = stored.get("best_episode_history", [])
            if training_history:
                from ..policy_rendering import PolicyRenderer  # noqa: PLC0415
                from ..policy_rendering import save_gif, PolicyRenderResult  # noqa: PLC0415
                from ..policy_rendering import _hashable_obs  # noqa: PLC0415

                policy = {_hashable_obs(obs): act for obs, act in training_history}
                render_env = factory.create(render_mode="rgb_array")
                if use_lang_state:
                    render_env = _LangStateEnv(render_env, translator=translator, env_id=env_id)
                renderer = PolicyRenderer(env=render_env, policy=policy)
                frames = renderer.run(max_steps=max_steps, seed=seed)

                n_gif_frames = save_gif(frames, gif_path, fps=fps, annotate=True)
                if n_gif_frames == 0:
                    return (
                        f"Policy replay completed but no frames were captured.\n"
                        f"Environment '{env_id}' may not support rgb_array rendering."
                    )

                agent_type = stored.get("agent_type", "unknown")
                summary = (
                    f"PolicyRenderResult\n"
                    f"  Agent type:        {agent_type}\n"
                    f"  Environment:       {env_id}\n"
                    f"  Source:            best training episode ({len(training_history)} steps)\n"
                    f"  Frames rendered:   {len(frames)}\n"
                    f"  GIF saved:         {n_gif_frames} frames  \u2192 {gif_path}"
                )
                gif_bytes = Path(gif_path).read_bytes()
                b64 = base64.b64encode(gif_bytes).decode("ascii")
                return f"{summary}\n\nSaved to: {gif_path}\n\ndata:image/gif;base64,{b64}"

            # Fallback: no stored history — run greedy evaluation episodes
            agent = stored["agent"]

            if use_lang_state:
                train_env = _LangStateEnv(train_env, translator=translator, env_id=env_id)

            def _greedy_fn(obs: Any) -> Any:
                if hasattr(agent, "act_greedy"):
                    return agent.act_greedy(obs)
                return agent.act(obs)

            base = GreedyEpisodeProtocol(
                policy_fn=_greedy_fn,
                max_steps=max_steps,
                seed=seed,
                record_history=True,
            )
            protocol = MultiEpisodeProtocol(base, n_episodes=n_episodes, base_seed=seed)
        else:
            base = RandomEpisodeProtocol(
                max_steps=max_steps,
                seed=seed,
                record_history=True,
            )
            protocol = MultiEpisodeProtocol(base, n_episodes=n_episodes, base_seed=seed)

        result = protocol(train_env)
    finally:
        train_env.close()

    # ── Render best episode to GIF ────────────────────────────────────────────
    try:
        render_result = render_optimal_policy(
            result,
            env_factory=factory,
            render_mode="rgb_array",
            max_steps=max_steps,
            seed=seed,
            output_gif=gif_path,
            gif_fps=fps,
            gif_annotate=True,
        )
    except Exception as exc:
        return f"Rendering failed: {exc}"

    if render_result.n_gif_frames == 0:
        return (
            f"Policy replay completed but no frames were captured.\n"
            f"Environment '{env_id}' may not support rgb_array rendering.\n\n"
            f"{render_result}"
        )

    # ── Return GIF as data URL so Claude can display it ───────────────────────
    gif_bytes = Path(gif_path).read_bytes()
    b64 = base64.b64encode(gif_bytes).decode("ascii")

    return (
        f"{render_result}\n\n"
        f"Saved to: {gif_path}\n\n"
        f"data:image/gif;base64,{b64}"
    )


# ── Static path-image tool ────────────────────────────────────────────────────

@mcp.tool()
def rl_render_policy_image(
    env_id: str,
    n_episodes: int = 30,
    max_steps: int = 200,
    seed: Optional[int] = None,
    agent_id: str = "",
    max_cols: int = 8,
    thumb_width: int = 120,
) -> str:
    """
    Render the optimal policy as a single static PNG image showing all steps
    tiled in a grid.

    NOTE: For the standard single-image render, prefer rl_render_policy_overlay
    instead.  Use this tool when you want every step visible as a separate
    labelled thumbnail in a grid layout.

    Runs several episodes, picks the best one, replays it with rendering
    enabled, and tiles every frame into a compact grid PNG.  Each thumbnail
    shows the step number and reward; terminal frames have a red border and
    sub-goal frames a green border.

    The image is returned as an inline data URL (``data:image/png;base64,…``)
    so Claude can display it directly, and is also saved locally to
    ``<cwd>/rlip_results/<env>/renders/``.

    Parameters
    ----------
    env_id:
        Environment to render (must be visible in rl_list_environments).
    n_episodes:
        Number of episodes to collect before selecting the best policy.
    max_steps:
        Hard cap on each episode length.
    seed:
        Random seed for reproducibility.
    agent_id:
        Optional ID of a trained agent (from rl_train_agent).  When
        provided, the agent's greedy policy is used instead of random
        exploration.
    max_cols:
        Maximum number of thumbnail columns in the grid (default 8).
    thumb_width:
        Width of each thumbnail in pixels (default 120).

    Returns the path image as a data URL (``data:image/png;base64,...``) that
    Claude can display, plus the local file path.
    """
    from datetime import datetime

    from ..interaction_protocols import (
        MultiEpisodeProtocol,
        RandomEpisodeProtocol,
        GreedyEpisodeProtocol,
    )
    from ..policy_rendering import (
        render_optimal_policy,
        PolicyRenderer,
        save_path_image,
        PolicyRenderResult,
        _hashable_obs,
    )

    # ── Resolve environment factory ───────────────────────────────────────────
    if not _registry or env_id not in _registry:
        return f"Unknown environment '{env_id}'. Use rl_list_environments() to see available envs."

    factory = _registry.get(env_id)

    # ── Determine PNG output path ─────────────────────────────────────────────
    env_render_dir = _env_renders_dir(env_id)
    env_render_dir.mkdir(parents=True, exist_ok=True)
    safe_id = env_id.replace("/", "_").replace("-", "_").replace(" ", "_")
    _stored = _trained_agents.get(agent_id) if agent_id else None
    agent_type_label = _stored.get("agent_type", "unknown") if _stored else "random"
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = env_render_dir / f"{safe_id}_{agent_type_label}_{n_episodes}ep_{_ts}_path.png"

    # ── Collect episodes ──────────────────────────────────────────────────────
    train_env = factory.create(render_mode=None)
    frames = None
    summary = None

    try:
        if agent_id and agent_id in _trained_agents:
            stored = _trained_agents[agent_id]
            use_lang_state = bool(stored.get("use_language_state", False))
            translator = None
            if use_lang_state:
                from ..language_translation import get_translator  # noqa: PLC0415

                translator = _custom_translators.get(env_id) or get_translator(env_id)
                if translator is None:
                    return (
                        f"Agent '{agent_id}' was trained with use_language_state=True, "
                        f"but no translator is registered for '{env_id}'.\n"
                        "Register one with rl_set_translator_code() first."
                    )

            training_history = stored.get("best_episode_history", [])
            if training_history:
                policy = {_hashable_obs(obs): act for obs, act in training_history}
                render_env = factory.create(render_mode="rgb_array")
                if use_lang_state:
                    render_env = _LangStateEnv(render_env, translator=translator, env_id=env_id)
                renderer = PolicyRenderer(env=render_env, policy=policy)
                frames = renderer.run(max_steps=max_steps, seed=seed)
                agent_type = stored.get("agent_type", "unknown")
                summary = (
                    f"PolicyRenderResult\n"
                    f"  Agent type:        {agent_type}\n"
                    f"  Environment:       {env_id}\n"
                    f"  Source:            best training episode ({len(training_history)} steps)\n"
                    f"  Frames rendered:   {len(frames)}\n"
                )
            else:
                # Fallback: run greedy evaluation episodes
                agent = stored["agent"]
                if use_lang_state:
                    train_env = _LangStateEnv(train_env, translator=translator, env_id=env_id)

                def _greedy_fn(obs: Any) -> Any:
                    if hasattr(agent, "act_greedy"):
                        return agent.act_greedy(obs)
                    return agent.act(obs)

                base = GreedyEpisodeProtocol(
                    policy_fn=_greedy_fn,
                    max_steps=max_steps,
                    seed=seed,
                    record_history=True,
                )
                protocol = MultiEpisodeProtocol(base, n_episodes=n_episodes, base_seed=seed)
                result = protocol(train_env)
        else:
            base = RandomEpisodeProtocol(
                max_steps=max_steps,
                seed=seed,
                record_history=True,
            )
            protocol = MultiEpisodeProtocol(base, n_episodes=n_episodes, base_seed=seed)
            result = protocol(train_env)
    finally:
        train_env.close()

    # ── If frames not yet collected, use render_optimal_policy ────────────────
    if frames is None:
        try:
            render_result = render_optimal_policy(
                result,
                env_factory=factory,
                render_mode="rgb_array",
                max_steps=max_steps,
                seed=seed,
                output_path_image=png_path,
                path_image_max_cols=max_cols,
                path_image_thumb_width=thumb_width,
            )
        except Exception as exc:
            return f"Rendering failed: {exc}"

        if not Path(png_path).exists():
            return (
                f"Policy replay completed but no frames were captured.\n"
                f"Environment '{env_id}' may not support rgb_array rendering.\n\n"
                f"{render_result}"
            )

        png_bytes = Path(png_path).read_bytes()
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return (
            f"{render_result}\n\n"
            f"Saved to: {png_path}\n\n"
            f"data:image/png;base64,{b64}"
        )

    # ── Save path image from pre-collected frames ─────────────────────────────
    saved = save_path_image(
        frames,
        png_path,
        max_cols=max_cols,
        thumb_width=thumb_width,
    )
    if not saved:
        return (
            f"Policy replay completed but no frames were captured.\n"
            f"Environment '{env_id}' may not support rgb_array rendering."
        )

    png_bytes = Path(png_path).read_bytes()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return (
        f"{summary}"
        f"  Path image saved:  {png_path}\n\n"
        f"Saved to: {png_path}\n\n"
        f"data:image/png;base64,{b64}"
    )


# ── Overlay composite tool ────────────────────────────────────────────────────

@mcp.tool()
def rl_render_policy_overlay(
    env_id: str,
    n_episodes: int = 30,
    max_steps: int = 200,
    seed: Optional[int] = None,
    agent_id: str = "",
) -> str:
    """
    STANDARD RENDER METHOD — use this by default for all policy visualisation.

    Renders the optimal policy as a single composite PNG where every frame is
    overlaid at equal opacity, showing all visited states simultaneously in
    one static image.

    Frames are blended using an arithmetic mean so each visited state is
    equally visible.  A subtle blue-to-red tint gradient encodes trajectory
    direction (early steps are cooler, later steps warmer).

    This is the standard single-image render method — use it for a compact,
    animation-free view of where the policy travels across the full episode.

    The image is returned as an inline data URL (``data:image/png;base64,…``)
    so Claude can display it directly, and is also saved locally to
    ``<cwd>/rlip_results/<env>/renders/``.

    Parameters
    ----------
    env_id:
        Environment to render (must be visible in rl_list_environments).
    n_episodes:
        Number of episodes to collect before selecting the best policy.
    max_steps:
        Hard cap on each episode length.
    seed:
        Random seed for reproducibility.
    agent_id:
        Optional ID of a trained agent (from rl_train_agent).  When
        provided, the agent's greedy policy is used instead of random
        exploration.

    Returns the overlay image as a data URL (``data:image/png;base64,...``)
    that Claude can display, plus the local file path.
    """
    from datetime import datetime

    from ..interaction_protocols import (
        MultiEpisodeProtocol,
        RandomEpisodeProtocol,
        GreedyEpisodeProtocol,
    )
    from ..policy_rendering import (
        render_optimal_policy,
        PolicyRenderer,
        save_overlay_image,
        _hashable_obs,
    )

    # ── Resolve environment factory ───────────────────────────────────────────
    if not _registry or env_id not in _registry:
        return f"Unknown environment '{env_id}'. Use rl_list_environments() to see available envs."

    factory = _registry.get(env_id)

    # ── Determine PNG output path ─────────────────────────────────────────────
    env_render_dir = _env_renders_dir(env_id)
    env_render_dir.mkdir(parents=True, exist_ok=True)
    safe_id = env_id.replace("/", "_").replace("-", "_").replace(" ", "_")
    _stored = _trained_agents.get(agent_id) if agent_id else None
    agent_type_label = _stored.get("agent_type", "unknown") if _stored else "random"
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = env_render_dir / f"{safe_id}_{agent_type_label}_{n_episodes}ep_{_ts}_overlay.png"

    # ── Collect episodes ──────────────────────────────────────────────────────
    train_env = factory.create(render_mode=None)
    frames = None
    summary = None

    try:
        if agent_id and agent_id in _trained_agents:
            stored = _trained_agents[agent_id]
            use_lang_state = bool(stored.get("use_language_state", False))
            translator = None
            if use_lang_state:
                from ..language_translation import get_translator  # noqa: PLC0415

                translator = _custom_translators.get(env_id) or get_translator(env_id)
                if translator is None:
                    return (
                        f"Agent '{agent_id}' was trained with use_language_state=True, "
                        f"but no translator is registered for '{env_id}'.\n"
                        "Register one with rl_set_translator_code() first."
                    )

            training_history = stored.get("best_episode_history", [])
            if training_history:
                policy = {_hashable_obs(obs): act for obs, act in training_history}
                render_env = factory.create(render_mode="rgb_array")
                if use_lang_state:
                    render_env = _LangStateEnv(render_env, translator=translator, env_id=env_id)
                renderer = PolicyRenderer(env=render_env, policy=policy)
                frames = renderer.run(max_steps=max_steps, seed=seed)
                agent_type = stored.get("agent_type", "unknown")
                summary = (
                    f"PolicyRenderResult\n"
                    f"  Agent type:        {agent_type}\n"
                    f"  Environment:       {env_id}\n"
                    f"  Source:            best training episode ({len(training_history)} steps)\n"
                    f"  Frames rendered:   {len(frames)}\n"
                )
            else:
                agent = stored["agent"]
                if use_lang_state:
                    train_env = _LangStateEnv(train_env, translator=translator, env_id=env_id)

                def _greedy_fn(obs: Any) -> Any:
                    if hasattr(agent, "act_greedy"):
                        return agent.act_greedy(obs)
                    return agent.act(obs)

                base = GreedyEpisodeProtocol(
                    policy_fn=_greedy_fn,
                    max_steps=max_steps,
                    seed=seed,
                    record_history=True,
                )
                protocol = MultiEpisodeProtocol(base, n_episodes=n_episodes, base_seed=seed)
                result = protocol(train_env)
        else:
            base = RandomEpisodeProtocol(
                max_steps=max_steps,
                seed=seed,
                record_history=True,
            )
            protocol = MultiEpisodeProtocol(base, n_episodes=n_episodes, base_seed=seed)
            result = protocol(train_env)
    finally:
        train_env.close()

    # ── If frames not yet collected, use render_optimal_policy ────────────────
    if frames is None:
        try:
            render_result = render_optimal_policy(
                result,
                env_factory=factory,
                render_mode="rgb_array",
                max_steps=max_steps,
                seed=seed,
                output_overlay_image=png_path,
            )
        except Exception as exc:
            return f"Rendering failed: {exc}"

        if not Path(png_path).exists():
            return (
                f"Policy replay completed but no frames were captured.\n"
                f"Environment '{env_id}' may not support rgb_array rendering.\n\n"
                f"{render_result}"
            )

        png_bytes = Path(png_path).read_bytes()
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return (
            f"{render_result}\n\n"
            f"Saved to: {png_path}\n\n"
            f"data:image/png;base64,{b64}"
        )

    # ── Save overlay from pre-collected frames ────────────────────────────────
    saved = save_overlay_image(frames, png_path)
    if not saved:
        return (
            f"Policy replay completed but no frames were captured.\n"
            f"Environment '{env_id}' may not support rgb_array rendering."
        )

    png_bytes = Path(png_path).read_bytes()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return (
        f"{summary}"
        f"  Overlay image saved: {png_path}\n\n"
        f"Saved to: {png_path}\n\n"
        f"data:image/png;base64,{b64}"
    )
