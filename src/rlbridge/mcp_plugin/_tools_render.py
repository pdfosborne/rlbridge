"""
Rendering MCP tools and resources for the rlbridge plugin.

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

@mcp.resource("rlbridge://environments")
def environments_resource() -> str:
    """All registered rlbridge environments as JSON."""
    from ..protocol.constants import Methods
    result = _dispatch(Methods.LIST_ENVIRONMENTS, tags=[], namespace=None)
    return json.dumps(result, indent=2)


@mcp.resource("rlbridge://instances")
def instances_resource() -> str:
    """All active rlbridge environment instances as JSON."""
    from ..protocol.constants import Methods
    result = _dispatch(Methods.LIST_INSTANCES)
    return json.dumps(result, indent=2)


# ── Render tool ───────────────────────────────────────────────────────────────

@mcp.tool()
def rl_render_policy(
    env_id: str,
    n_episodes: int = 100,
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

    # ── Collect evaluation episodes ───────────────────────────────────────────
    eval_env = factory.create(render_mode=None)
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
                eval_env = _LangStateEnv(eval_env, translator=translator, env_id=env_id)

            agent = stored["agent"]

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

        result = protocol(eval_env)
    finally:
        eval_env.close()

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
    n_episodes: int = 100,
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

    # ── Collect evaluation episodes ───────────────────────────────────────────
    eval_env = factory.create(render_mode=None)
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
                eval_env = _LangStateEnv(eval_env, translator=translator, env_id=env_id)

            agent = stored["agent"]

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

        result = protocol(eval_env)
    finally:
        eval_env.close()

    # ── Render best episode to path image ─────────────────────────────────────
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


# ── Overlay composite tool ────────────────────────────────────────────────────

@mcp.tool()
def rl_render_policy_overlay(
    env_id: str,
    n_episodes: int = 100,
    max_steps: int = 200,
    seed: Optional[int] = None,
    agent_id: str = "",
) -> str:
    """
    STANDARD RENDER METHOD - use this by default for all policy visualisation.

    Renders the optimal policy as a single composite PNG where every frame is
    overlaid at equal opacity, showing all visited states simultaneously in
    one static image.

    Frames are blended using an arithmetic mean so each visited state is
    equally visible.  A subtle blue-to-red tint gradient encodes trajectory
    direction (early steps are cooler, later steps warmer).

    This is the standard single-image render method - use it for a compact,
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

    # ── Collect evaluation episodes ───────────────────────────────────────────
    eval_env = factory.create(render_mode=None)
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
                eval_env = _LangStateEnv(eval_env, translator=translator, env_id=env_id)

            agent = stored["agent"]

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

        result = protocol(eval_env)
    finally:
        eval_env.close()

    # ── Render best episode to overlay image ──────────────────────────────────
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
