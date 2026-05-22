"""
Basic environment interaction MCP tools for the rlbridge plugin.

Tools: rl_list_environments, rl_create, rl_reset, rl_step, rl_sample_action,
       rl_spaces, rl_render, rl_close, rl_list_instances, rl_run_episode,
       rl_get_suggested_hyperparameters, rl_update_suggested_hyperparameters.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from ._dispatch import _dispatch, _fmt_obs
from ._state import _hp_override_path, _hp_overrides, mcp


@mcp.tool()
def rl_list_environments(
    tags: str = "",
    namespace: str = "",
) -> str:
    """
    List all available RL environments registered with rlbridge.

    Parameters
    ----------
    tags:
        Comma-separated list of tags to filter by (e.g. "classic-control").
    namespace:
        Filter by namespace, e.g. "gymnasium".

    Returns a formatted list of environment IDs with metadata.
    """
    from ..protocol.constants import Methods
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    result = _dispatch(
        Methods.LIST_ENVIRONMENTS,
        tags=tag_list,
        namespace=namespace or None,
    )
    envs = result.get("environments", [])
    if not envs:
        return "No environments match the given filters."

    lines = [f"Found {len(envs)} environment(s):\n"]
    for env in envs:
        line = f"  • {env['env_id']}"
        if env.get("description"):
            line += f"  -  {env['description']}"
        if env.get("max_episode_steps"):
            line += f"  (max {env['max_episode_steps']} steps)"
        if env.get("reward_threshold") is not None:
            line += f"  [threshold: {env['reward_threshold']}]"
        lines.append(line)
    return "\n".join(lines)


@mcp.tool()
def rl_create(
    env_id: str,
    render_mode: str = "",
) -> str:
    """
    Create a new RL environment instance.

    Parameters
    ----------
    env_id:
        The Gymnasium environment ID, e.g. "CartPole-v1", "LunarLander-v3".
    render_mode:
        Optional render mode: "rgb_array" (returns PNG image), "ansi" (ASCII).
        Leave empty for no rendering.

    Returns the instance_id needed for all subsequent calls, plus space info.
    """
    from ..protocol.constants import Methods
    result = _dispatch(
        Methods.CREATE_ENVIRONMENT,
        env_id=env_id,
        render_mode=render_mode or None,
        kwargs={},
    )
    iid = result["instance_id"]
    obs_space = result.get("observation_space", {})
    act_space = result.get("action_space", {})

    return (
        f"Created environment: {env_id}\n"
        f"Instance ID: {iid}\n\n"
        f"Observation space: {json.dumps(obs_space, indent=2)}\n\n"
        f"Action space: {json.dumps(act_space, indent=2)}\n\n"
        f"Call rl_reset(instance_id='{iid}') to start an episode."
    )


@mcp.tool()
def rl_reset(
    instance_id: str,
    seed: Optional[int] = None,
) -> str:
    """
    Reset an RL environment to its initial state.

    Must be called before rl_step().  Returns the initial observation.

    Parameters
    ----------
    instance_id:
        The instance_id returned by rl_create().
    seed:
        Optional integer seed for reproducibility.
    """
    from ..protocol.constants import Methods
    result = _dispatch(Methods.RESET, instance_id=instance_id, seed=seed, options={})
    obs = result.get("observation")
    info = result.get("info", {})
    lines = [
        f"Environment reset (instance: {instance_id})",
        f"Initial observation: {_fmt_obs(obs)}",
    ]
    if info:
        lines.append(f"Info: {json.dumps(info, indent=2)}")
    lines.append("\nReady to step. Use rl_step() with an action from the action space.")
    return "\n".join(lines)


@mcp.tool()
def rl_step(
    instance_id: str,
    action: str,
) -> str:
    """
    Execute one action in the RL environment.

    Parameters
    ----------
    instance_id:
        The instance_id returned by rl_create().
    action:
        The action to execute.  For Discrete spaces use an integer (e.g. "1").
        For Box spaces use a JSON array (e.g. "[0.5, -0.3]").
        For Dict spaces use a JSON object.

    Returns the next observation, reward, done flags, and info dict.
    """
    from ..protocol.constants import Methods

    # Parse action from string
    try:
        parsed_action: Any = json.loads(action)
    except json.JSONDecodeError:
        # Might be a bare integer like "2"
        try:
            parsed_action = int(action)
        except ValueError:
            try:
                parsed_action = float(action)
            except ValueError:
                parsed_action = action

    result = _dispatch(Methods.STEP, instance_id=instance_id, action=parsed_action)
    obs = result.get("observation")
    reward = result.get("reward", 0.0)
    terminated = result.get("terminated", False)
    truncated = result.get("truncated", False)
    info = result.get("info", {})

    status = ""
    if terminated:
        status = "\nEpisode TERMINATED (natural end). Call rl_reset() to start again."
    elif truncated:
        status = "\nEpisode TRUNCATED (time limit). Call rl_reset() to start again."

    lines = [
        f"Step result (instance: {instance_id})",
        f"  Observation: {_fmt_obs(obs)}",
        f"  Reward:      {reward:+.4f}",
        f"  Terminated:  {terminated}",
        f"  Truncated:   {truncated}",
    ]
    if info:
        lines.append(f"  Info: {json.dumps(info)}")
    if status:
        lines.append(status)
    return "\n".join(lines)


@mcp.tool()
def rl_sample_action(instance_id: str) -> str:
    """
    Sample a random valid action from the environment's action space.

    Useful when exploring or debugging.

    Parameters
    ----------
    instance_id:
        The instance_id returned by rl_create().
    """
    from ..protocol.constants import Methods
    # We get the spaces and sample locally for efficiency
    result = _dispatch(Methods.GET_SPACES, instance_id=instance_id)
    act_space_desc = result.get("action_space", {})

    space_type = act_space_desc.get("type", "")
    if space_type == "Discrete":
        import random
        n = act_space_desc["n"]
        start = act_space_desc.get("start", 0)
        action = random.randint(start, start + n - 1)
    else:
        # For complex spaces, fall back to the environment's own sampler
        try:
            import gymnasium as gym
            from ..environments.utils import description_to_space
            from ..protocol.messages import SpaceDescription
            from pydantic import TypeAdapter
            ta = TypeAdapter(SpaceDescription)
            space = description_to_space(ta.validate_python(act_space_desc))
            import numpy as np
            from ..environments.utils import numpy_to_python
            action = numpy_to_python(space.sample())
        except Exception:
            action = "Unable to sample - use rl_step with an action from the action space description"

    return f"Sampled action: {json.dumps(action)}\nUse with: rl_step(instance_id='{instance_id}', action='{json.dumps(action)}')"


@mcp.tool()
def rl_spaces(instance_id: str) -> str:
    """
    Describe the observation and action spaces of an environment instance.

    Parameters
    ----------
    instance_id:
        The instance_id returned by rl_create().
    """
    from ..protocol.constants import Methods
    result = _dispatch(Methods.GET_SPACES, instance_id=instance_id)
    obs_space = result.get("observation_space", {})
    act_space = result.get("action_space", {})
    return (
        f"Spaces for instance {instance_id}:\n\n"
        f"Observation space:\n{json.dumps(obs_space, indent=2)}\n\n"
        f"Action space:\n{json.dumps(act_space, indent=2)}"
    )


@mcp.tool()
def rl_render(instance_id: str) -> str:
    """
    Render the current state of the environment.

    For rgb_array mode: returns a base64-encoded PNG you can view as an image.
    For ansi mode: returns ASCII art text.

    Parameters
    ----------
    instance_id:
        The instance_id returned by rl_create().
    """
    from ..protocol.constants import Methods
    result = _dispatch(Methods.RENDER, instance_id=instance_id)
    mode = result.get("mode", "unknown")

    if mode == "rgb_array" and result.get("data"):
        w = result.get("width", "?")
        h = result.get("height", "?")
        b64 = result["data"]
        # Return as a data URL for easy display
        return (
            f"Render ({w}×{h} PNG image):\n"
            f"data:image/png;base64,{b64}"
        )
    if mode == "ansi" and result.get("text"):
        return f"Render (ANSI):\n{result['text']}"

    return f"Render mode '{mode}': {json.dumps(result)}"


@mcp.tool()
def rl_close(instance_id: str) -> str:
    """
    Close and destroy an RL environment instance, freeing its resources.

    Parameters
    ----------
    instance_id:
        The instance_id returned by rl_create().
    """
    from ..protocol.constants import Methods
    result = _dispatch(Methods.CLOSE, instance_id=instance_id)
    closed = result.get("closed", False)
    if closed:
        return f"Environment instance {instance_id} closed successfully."
    return f"Instance {instance_id} was not found (may have already been closed)."


@mcp.tool()
def rl_list_instances() -> str:
    """
    List all currently active RL environment instances managed by rlbridge.

    Returns instance IDs, environment IDs, and initialization status.
    """
    from ..protocol.constants import Methods
    result = _dispatch(Methods.LIST_INSTANCES)
    instances = result.get("instances", [])
    total = result.get("total", 0)
    if total == 0:
        return "No active environment instances. Use rl_create() to start one."

    lines = [f"{total} active instance(s):\n"]
    for inst in instances:
        init_status = "ready" if inst.get("is_initialized") else "needs reset"
        lines.append(
            f"  • {inst['instance_id'][:8]}…  env={inst['env_id']}  "
            f"render={inst.get('render_mode') or 'none'}  status={init_status}"
        )
    return "\n".join(lines)


@mcp.tool()
def rl_run_episode(
    env_id: str,
    max_steps: int = 200,
    seed: Optional[int] = None,
    policy: str = "random",
) -> str:
    """
    Run a complete episode in an RL environment and report the results.

    This is a convenience tool that creates an env, resets it, runs until done
    (or max_steps), and returns a summary.

    Parameters
    ----------
    env_id:
        Gymnasium environment ID, e.g. "CartPole-v1".
    max_steps:
        Maximum number of steps before truncating the episode.
    seed:
        Optional integer seed.
    policy:
        Currently only "random" is supported (samples random actions).
    """
    from ..protocol.constants import Methods
    from pydantic import TypeAdapter
    from ..protocol.messages import SpaceDescription
    from ..environments.utils import description_to_space, numpy_to_python
    import random

    # Create
    create_result = _dispatch(
        Methods.CREATE_ENVIRONMENT,
        env_id=env_id,
        render_mode=None,
        kwargs={},
    )
    instance_id: str = create_result["instance_id"]

    try:
        act_space_desc = create_result["action_space"]

        # Reset
        _dispatch(Methods.RESET, instance_id=instance_id, seed=seed, options={})

        total_reward = 0.0
        step_count = 0

        # Build action sampler
        space_type = act_space_desc.get("type", "")
        if space_type == "Discrete":
            n = act_space_desc["n"]
            start = act_space_desc.get("start", 0)
            def sample_action() -> Any:
                return random.randint(start, start + n - 1)
        else:
            try:
                ta = TypeAdapter(SpaceDescription)
                space = description_to_space(ta.validate_python(act_space_desc))
                def sample_action() -> Any:   # type: ignore[misc]
                    return numpy_to_python(space.sample())
            except Exception:
                def sample_action() -> Any:   # type: ignore[misc]
                    return 0  # fallback

        # Run episode
        step_result: dict[str, Any] = {}
        for _ in range(max_steps):
            action = sample_action()
            step_result = _dispatch(
                Methods.STEP, instance_id=instance_id, action=action
            )
            total_reward += step_result.get("reward", 0.0)
            step_count += 1
            if step_result.get("terminated") or step_result.get("truncated"):
                break

        terminated = step_result.get("terminated", False)
        truncated = step_result.get("truncated", False)

    finally:
        _dispatch(Methods.CLOSE, instance_id=instance_id)

    end_reason = "terminated" if terminated else ("truncated" if truncated else "max_steps reached")
    return (
        f"Episode complete: {env_id}\n"
        f"  Policy:       {policy}\n"
        f"  Steps:        {step_count}\n"
        f"  Total reward: {total_reward:.4f}\n"
        f"  End reason:   {end_reason}\n"
        f"  Seed:         {seed if seed is not None else 'random'}"
    )


@mcp.tool()
def rl_get_suggested_hyperparameters(env_id: str) -> str:
    """
    Return the recommended training hyperparameters for a registered environment.

    Predefined rlbridge environments ship with environment-specific suggestions for
    agent type, episode counts, step limits, and RL hyperparameters.  Use these
    as a starting point before calling ``rl_train_agent()`` or
    ``rl_experiment_process()``.

    ``rl_experiment_process()`` automatically applies these suggestions when you
    do not override individual parameters, so calling this tool first is optional
    - it is mainly useful for inspection or when building a custom training call.

    Call ``rl_update_suggested_hyperparameters()`` after training runs to persist
    better values once convergence is observed.

    Parameters
    ----------
    env_id:
        A registered rlbridge environment ID, e.g. ``"Sailing-v0"``.

    Returns
    -------
    Formatted list of suggested hyperparameters, or a note that none are defined.
    """
    from ..environments.registry import registry as _env_registry  # noqa: PLC0415

    try:
        factory = _env_registry.get(env_id)
    except KeyError:
        return (
            f"Environment '{env_id}' is not registered.  "
            "Call rl_list_environments() to see available environments."
        )

    # Persisted override takes precedence over the factory default
    hp = _hp_overrides.get(env_id) or factory.env_info.suggested_hyperparameters
    source = "(persisted override)" if env_id in _hp_overrides else "(factory default)"
    if hp is None:
        return (
            f"No suggested hyperparameters are defined for '{env_id}'.\n"
            "You can pass hyperparameters explicitly to rl_experiment_process() "
            "or rl_train_agent(), then save the best values with "
            "rl_update_suggested_hyperparameters()."
        )

    lines = [
        f"Suggested hyperparameters for '{env_id}' {source}:\n",
        f"  agent_type:              {hp.agent_type}",
        f"  n_episodes:              {hp.n_episodes}",
        f"  max_steps:               {hp.max_steps}",
        "",
        "  Instruction matching:",
        f"    sub_goal_threshold:    {hp.sub_goal_threshold}",
        f"    top_k:                 {hp.top_k}",
        f"    min_episode_visits:    {hp.min_episode_visits}",
        "",
        "  Tabular-Q:",
        f"    alpha:                 {hp.alpha}",
        f"    gamma:                 {hp.gamma}",
        f"    epsilon:               {hp.epsilon}",
        f"    epsilon_min:           {hp.epsilon_min}",
        f"    epsilon_decay:         {hp.epsilon_decay}",
        "",
        "  DQN / PPO:",
        f"    hidden_size:           {hp.hidden_size}",
        f"    lr:                    {hp.lr}",
        "",
        f"Use: rl_experiment_process(agent_type='{hp.agent_type}', env_id='{env_id}')",
        "Parameters with matching values will be applied automatically.",
    ]
    return "\n".join(lines)


@mcp.tool()
def rl_update_suggested_hyperparameters(
    env_id: str,
    agent_type: Optional[str] = None,
    n_episodes: Optional[int] = None,
    max_steps: Optional[int] = None,
    sub_goal_threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    min_episode_visits: Optional[int] = None,
    alpha: Optional[float] = None,
    gamma: Optional[float] = None,
    epsilon: Optional[float] = None,
    epsilon_min: Optional[float] = None,
    epsilon_decay: Optional[float] = None,
    hidden_size: Optional[int] = None,
    lr: Optional[float] = None,
) -> str:
    """
    Update and persist the suggested training hyperparameters for an environment.

    Use this after completing training runs to record better hyperparameter values
    - in particular the **minimum number of episodes needed for convergence**.
    Only pass the fields you want to change; all others are left at their current
    suggested values.

    The updated values are saved to
    ``~/.rlbridge/environments/<env_id>/suggested_hyperparameters.json`` and will
    be loaded automatically in future sessions.  ``rl_experiment_process()``
    picks them up immediately.

    **Convergence guidance** - set ``n_episodes`` to the *lowest* episode
    count at which training reliably converges on this environment, not a
    conservative upper bound.
    This keeps experiment runs fast and avoids wasted compute on subsequent calls.

    Parameters
    ----------
    env_id:
        A registered rlbridge environment ID, e.g. ``"Sailing-v0"``.
    agent_type:
        Best agent type found (``"tabular_q"``, ``"dqn"``, or ``"ppo"``).
    n_episodes:
        Minimum episodes for reliable convergence.
    max_steps:
        Step cap per episode.
    sub_goal_threshold / top_k / min_episode_visits:
        Instruction-matching tuning.
    alpha / gamma / epsilon / epsilon_min / epsilon_decay:
        Best tabular-Q values found.
    hidden_size / lr:
        Best DQN/PPO values found.

    Returns
    -------
    Confirmation showing the full updated parameter set.
    """
    import json as _json  # noqa: PLC0415
    from ..environments.registry import registry as _env_registry  # noqa: PLC0415
    from ..protocol.messages import SuggestedHyperparameters  # noqa: PLC0415

    try:
        factory = _env_registry.get(env_id)
    except KeyError:
        return (
            f"Environment '{env_id}' is not registered.  "
            "Call rl_list_environments() to see available environments."
        )

    # Start from the best existing values: persisted override → factory default → bare defaults
    existing = _hp_overrides.get(env_id) or factory.env_info.suggested_hyperparameters
    base = existing.model_dump() if existing is not None else {}

    # Apply only the fields explicitly provided (not None)
    updates: dict[str, Any] = {}
    _fields = {
        "agent_type": agent_type,
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "sub_goal_threshold": sub_goal_threshold,
        "top_k": top_k,
        "min_episode_visits": min_episode_visits,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "hidden_size": hidden_size,
        "lr": lr,
    }
    for field, val in _fields.items():
        if val is not None:
            updates[field] = val

    if not updates:
        return (
            f"No fields were provided - nothing to update for '{env_id}'.\n"
            "Pass at least one parameter to change."
        )

    merged = {**base, **updates}
    new_hp = SuggestedHyperparameters(**merged)

    # Persist to disk
    hp_path = _hp_override_path(env_id)
    hp_path.parent.mkdir(parents=True, exist_ok=True)
    payload = new_hp.model_dump()
    payload["_env_id"] = env_id  # store env_id so the loader can recover it
    hp_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")

    # Update live cache
    _hp_overrides[env_id] = new_hp

    changed_lines = [f"  {k}: {base.get(k, '(new)')} → {v}" for k, v in updates.items()]
    return (
        f"Suggested hyperparameters updated for '{env_id}'.\n\n"
        f"Changed fields:\n"
        + "\n".join(changed_lines)
        + f"\n\nFull updated values:\n"
        f"  agent_type:              {new_hp.agent_type}\n"
        f"  n_episodes:              {new_hp.n_episodes}\n"
        f"  max_steps:               {new_hp.max_steps}\n"
        f"  sub_goal_threshold:      {new_hp.sub_goal_threshold}\n"
        f"  top_k:                   {new_hp.top_k}\n"
        f"  min_episode_visits:      {new_hp.min_episode_visits}\n"
        f"  alpha:                   {new_hp.alpha}\n"
        f"  gamma:                   {new_hp.gamma}\n"
        f"  epsilon:                 {new_hp.epsilon}\n"
        f"  epsilon_min:             {new_hp.epsilon_min}\n"
        f"  epsilon_decay:           {new_hp.epsilon_decay}\n"
        f"  hidden_size:             {new_hp.hidden_size}\n"
        f"  lr:                      {new_hp.lr}\n\n"
        f"Saved to: {hp_path}\n"
        f"rl_experiment_process() will use these values automatically."
    )
