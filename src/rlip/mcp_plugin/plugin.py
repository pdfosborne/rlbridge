"""
RLIP MCP Plugin for Claude Code
=================================
Exposes RL environments as MCP tools so Claude Code can interact with
Gymnasium environments directly via natural-language agent tasks.

How it works
------------
1. This file is an MCP server using FastMCP (stdio transport).
2. Claude Code launches it as a subprocess and communicates over stdin/stdout.
3. The plugin can run environments *in-process* (default) or *proxy* to a
   running RLIP HTTP server (set RLIP_SERVER_URL env var).

Claude Code configuration  (~/.claude.json)
-------------------------------------------
    {
      "mcpServers": {
        "rlip": {
          "command": "python",
          "args": ["-m", "rlip.mcp_plugin"],
          "type": "stdio"
        }
      }
    }

Or if you installed via pip as a script:
    {
      "mcpServers": {
        "rlip": {
          "command": "rlip-mcp",
          "type": "stdio"
        }
      }
    }
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)

# ── Select in-process vs proxy mode ──────────────────────────────────────────

_RLIP_SERVER_URL = os.environ.get("RLIP_SERVER_URL", "")

if _RLIP_SERVER_URL:
    # Proxy mode – forward calls to a running RLIP HTTP server
    from ..transport.http_client import RLIPClient as _RLIPClient
    _proxy = _RLIPClient(_RLIP_SERVER_URL)
    _in_process = False
else:
    # In-process mode – run environments directly in this process
    from ..environments.registry import registry as _registry
    from ..server.dispatcher import RLIPDispatcher as _Dispatcher
    from ..server.session import SessionManager as _SessionManager
    _session = _SessionManager(max_instances=int(os.environ.get("RLIP_MAX_INSTANCES", "16")))
    _dispatcher = _Dispatcher(registry=_registry, session=_session)
    _in_process = True


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP(
    "RLIP",
    instructions=(
        "RLIP gives you full control over reinforcement learning environments. "
        "Use rl_list_environments to browse available envs, rl_create to instantiate "
        "one, rl_reset to start an episode, rl_step to act, and rl_close when done. "
        "Always reset before the first step. Use rl_render to visualise the state.\n\n"
        "Custom environment workflow:\n"
        "1. rl_build_environment(env_id, gym_env_id, description, tags) – wrap any "
        "Gymnasium environment with custom metadata and cache it locally.\n"
        "2. rl_sample_states_for_translation(env_id) – sample unique states so you "
        "can write a language translator.\n"
        "3. rl_set_translator_code(env_id, python_code) – install and persist a "
        "translate() function you write for the environment.\n"
        "4. rl_translate_state(env_id, state) – test translation quality.\n"
        "5. rl_load_cached_environments() – restore all cached environments at startup.\n\n"
        "Instruction-following workflow:\n"
        "1. rl_match_instruction(env_id, instruction) – explore the environment and "
        "find the observed state whose language description best matches your goal. "
        "Returns a match_id.\n"
        "2. rl_instruction_run_episode(match_id) – run a sub-goal-shaped training "
        "episode where the matched state provides a bonus reward signal.\n\n"
        "RL agent training workflow:\n"
        "1. rl_list_agents() – see available agent types (tabular_q, dqn, ppo) "
        "and guidance on when to use each.\n"
        "2. rl_train_agent(agent_type, env_id, n_episodes) – train the chosen agent "
        "and get back an agent_id.\n"
        "   • Add use_language_state=True to train on language descriptions of "
        "observations instead of raw numeric values.  Ideal for tabular_q with "
        "environments that have a registered translator (e.g. Sailing-v0).\n"
        "3. rl_run_agent_episode(agent_id) – evaluate the trained agent for one episode "
        "(automatically uses the same obs mode as training).\n"
        "4. rl_render_policy(env_id, agent_id=agent_id) – render the best training episode as a GIF.\n\n"
        "Combined instruction-following + agent training workflow:\n"
        "1. rl_match_instruction(env_id, instruction) – explore and find the sub-goal state. "
        "Returns a match_id.\n"
        "2. rl_train_agent(agent_type, env_id, match_id=match_id, use_language_state=True) – "
        "train with sub-goal reward shaping AND language observations (use_language_state is "
        "optional but recommended when a translator is available).\n"
        "3. rl_render_policy(env_id, agent_id=agent_id) – render the result."
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dispatch(method: str, **params: Any) -> Any:
    """Route a call to either the in-process dispatcher or remote proxy."""
    if _in_process:
        request = {"jsonrpc": "2.0", "id": "mcp", "method": method, "params": params}
        response = _dispatcher.dispatch(request)
        if response.get("error"):
            err = response["error"]
            raise RuntimeError(f"[{err['code']}] {err['message']}")
        return response.get("result", {})
    else:
        # Proxy – map method name to client method
        _proxy_dispatch(method, params)


def _proxy_dispatch(method: str, params: dict[str, Any]) -> Any:
    """Dispatch to the remote HTTP proxy client."""
    from ..protocol.constants import Methods
    mapping = {
        Methods.INITIALIZE:         lambda p: _proxy.initialize(**p),
        Methods.LIST_ENVIRONMENTS:  lambda p: _proxy.list_environments(**p).model_dump(),
        Methods.CREATE_ENVIRONMENT: lambda p: _proxy.create_environment(**p).model_dump(),
        Methods.RESET:              lambda p: _proxy.reset(**p).model_dump(),
        Methods.STEP:               lambda p: _proxy.step(**p).model_dump(),
        Methods.GET_SPACES:         lambda p: _proxy.get_spaces(**p).model_dump(),
        Methods.RENDER:             lambda p: _proxy.render(**p).model_dump(),
        Methods.CLOSE:              lambda p: _proxy.close_environment(**p).model_dump(),
        Methods.LIST_INSTANCES:     lambda p: _proxy.list_instances().model_dump(),
    }
    fn = mapping.get(method)
    if fn is None:
        raise ValueError(f"Unknown method: {method}")
    return fn(params)


def _fmt_obs(obs: Any) -> str:
    """Format an observation for readable MCP tool output."""
    if isinstance(obs, list):
        if len(obs) <= 16:
            formatted = ", ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in obs)
            return f"[{formatted}]"
        return f"<array of {len(obs)} values, first 8: {obs[:8]}>"
    return json.dumps(obs, indent=2)


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def rl_list_environments(
    tags: str = "",
    namespace: str = "",
) -> str:
    """
    List all available RL environments registered with RLIP.

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
            line += f"  –  {env['description']}"
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
            action = "Unable to sample – use rl_step with an action from the action space description"

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
    List all currently active RL environment instances managed by RLIP.

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
        reset_result = _dispatch(
            Methods.RESET, instance_id=instance_id, seed=seed, options={}
        )

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


# ── Instruction-following tools ───────────────────────────────────────────────

# Keep a small in-process cache of InstructionFollowingProtocol instances so
# that rl_instruction_run_episode can re-use a previously matched sub-goal
# without re-running exploration every time.
_instruction_protocols: dict[str, Any] = {}


@mcp.tool()
def rl_clear_obs_cache(env_id: str = "") -> str:
    """
    Clear the in-memory observation corpus cache used by rl_match_instruction.

    Call this when you have updated a language translator and want
    rl_match_instruction to rebuild the corpus with fresh translations,
    or when you want to force re-exploration of the state space.

    Parameters
    ----------
    env_id:
        Clear only the cache for this environment ID.
        When empty (default) the entire cache is cleared for all environments.

    Returns
    -------
    Confirmation of what was cleared.
    """
    from ..instruction_following import clear_obs_cache, obs_cache_info
    from ..language_translation.caching import clear_translation_cache, translation_cache_info

    before_obs = obs_cache_info()
    before_trans = translation_cache_info()

    target = env_id.strip() or None
    clear_obs_cache(target)
    clear_translation_cache(target)

    scope = f"'{target}'" if target else "all environments"
    obs_cleared = sum(before_obs[k] for k in (([target] if target else list(before_obs.keys()))) if k in before_obs)
    trans_cleared = sum(before_trans[k] for k in (([target] if target else list(before_trans.keys()))) if k in before_trans)

    return (
        f"Cache cleared for {scope}.\n"
        f"  Observation corpus entries removed: {obs_cleared}\n"
        f"  Translation cache entries removed:  {trans_cleared}\n\n"
        "Next call to rl_match_instruction will run fresh exploration."
    )


@mcp.tool()
def rl_match_instruction(
    env_id: str,
    instruction: str,
    exploration_steps: int = 100,
    seed: Optional[int] = None,
    top_k: int = 5,
) -> str:
    """
    Explore an RL environment, translate observed states to language, and
    find which observed state best matches a natural-language instruction
    using TF-IDF cosine similarity.

    This is the first step of instruction-following RL.  After calling this
    tool you can run rl_instruction_run_episode() to train with the matched
    state as a sub-goal.

    Parameters
    ----------
    env_id:
        A registered RLIP environment ID, e.g. "Sailing-v0".
    instruction:
        The natural-language goal to match, e.g.
        "sail towards the beach side".
    exploration_steps:
        Number of random steps used to build the observation corpus.
        More steps give broader coverage; 50–200 is usually sufficient.
    seed:
        Optional integer seed for reproducible exploration.
    top_k:
        Number of top-ranked matches to include in the output (max 10).

    Returns
    -------
    A text summary of the best-matched state, its similarity score, and
    a match_id you can pass to rl_instruction_run_episode().
    """
    import uuid
    from ..instruction_following import match_instruction, build_instruction_following_protocol
    from ..environments.registry import registry as _env_registry

    try:
        factory = _env_registry.get(env_id)
        env = factory.create()
    except KeyError:
        return (
            f"Environment '{env_id}' is not registered.  "
            "Call rl_list_environments() to see what is available."
        )

    try:
        match = match_instruction(
            instruction,
            env,
            seed=seed,
            max_steps=exploration_steps,
        )
    except ValueError as exc:
        return f"Instruction matching failed: {exc}"

    # Build and cache the ready-to-run protocol so the agent can immediately
    # call rl_instruction_run_episode without re-running exploration.
    protocol = build_instruction_following_protocol(
        instruction,
        env,
        seed=seed,
        max_steps=exploration_steps,
    )
    # The env was consumed by exploration; protocol will reset it on __call__.
    match_id = uuid.uuid4().hex[:12]
    _instruction_protocols[match_id] = {
        "protocol": protocol,
        "env_id":   env_id,
        "env":      env,
    }

    top_k = max(1, min(top_k, 10))
    top_lines = [
        f"  {i+1}. sim={sc:.4f}  {lg[:100]}"
        for i, (lg, sc) in enumerate(match.all_scores[:top_k])
    ]

    return (
        f"Instruction matched for '{env_id}':\n\n"
        f"  Instruction:   {instruction!r}\n"
        f"  Best match:    {match.matched_language}\n"
        f"  Similarity:    {match.similarity_score:.4f}\n"
        f"  Match ID:      {match_id}\n\n"
        f"Top {top_k} candidates:\n" + "\n".join(top_lines) + "\n\n"
        f"Use rl_instruction_run_episode(match_id='{match_id}') to run a "
        f"training episode with this state as a sub-goal."
    )


@mcp.tool()
def rl_instruction_run_episode(
    match_id: str,
    max_steps: int = 200,
    sub_goal_bonus: float = 1.0,
    sub_goal_threshold: float = 0.5,
    sub_goal_repeatable: bool = False,
    seed: Optional[int] = None,
) -> str:
    """
    Run a sub-goal-shaped RL training episode using a matched instruction.

    Must be called after rl_match_instruction().  The previously matched
    state acts as a language-grounded sub-goal: whenever the agent's
    observed state has a cosine similarity ≥ sub_goal_threshold to the
    sub-goal description, an additional bonus reward is added.

    Parameters
    ----------
    match_id:
        The match_id returned by rl_match_instruction().
    max_steps:
        Maximum steps for the training episode.
    sub_goal_bonus:
        Bonus reward added when the language similarity threshold is met.
    sub_goal_threshold:
        Cosine similarity threshold (0–1) required to award the bonus.
        Lower values make the sub-goal easier to reach.
    sub_goal_repeatable:
        False (default) – bonus awarded at most once per episode.
        True – bonus awarded on every step the threshold is met.
    seed:
        Optional seed for the training episode reset.

    Returns
    -------
    A detailed summary of the episode including total reward,
    which step(s) the sub-goal was reached, and a step-by-step
    language trajectory excerpt (up to 10 steps shown).
    """
    entry = _instruction_protocols.get(match_id)
    if entry is None:
        return (
            f"Match ID '{match_id}' not found.  "
            "Run rl_match_instruction() first to obtain a valid match_id."
        )

    protocol = entry["protocol"]
    env = entry["env"]

    # Apply per-call overrides
    protocol.max_steps = max_steps
    protocol.sub_goal_bonus = sub_goal_bonus
    protocol.sub_goal_threshold = sub_goal_threshold
    protocol.sub_goal_repeatable = sub_goal_repeatable
    if seed is not None:
        protocol.seed = seed

    result = protocol(env)
    ep = result.episodes[0]

    sub_goal_steps = [
        r.step for r in ep.history if r.info.get("sub_goal_reached")
    ]
    sim_values = [
        r.info.get("sub_goal_similarity")
        for r in ep.history
        if r.info.get("sub_goal_similarity") is not None
    ]
    max_sim = max(sim_values) if sim_values else None

    # Build a readable trajectory excerpt (first 10 steps)
    excerpt_lines: list[str] = []
    for rec in ep.history[:10]:
        sim = rec.info.get("sub_goal_similarity", "n/a")
        hit = " ◀ sub-goal" if rec.info.get("sub_goal_reached") else ""
        lang = (rec.language_obs or "")[:80]
        excerpt_lines.append(
            f"  step {rec.step:3d}  r={rec.reward:+.3f}  sim={sim}  {lang}{hit}"
        )
    if len(ep.history) > 10:
        excerpt_lines.append(f"  … ({len(ep.history) - 10} more steps not shown)")

    peak_line = f"\n  Peak similarity: {max_sim:.4f}" if max_sim is not None else ""

    return (
        f"Instruction-following episode complete\n"
        f"  Environment:   {result.env_id}\n"
        f"  Instruction:   {protocol.instruction!r}\n"
        f"  Sub-goal:      {protocol.sub_goal_language[:80]}\n"
        f"  Threshold:     {sub_goal_threshold}\n\n"
        f"  Steps:         {ep.steps}\n"
        f"  Total reward:  {ep.total_reward:.4f}\n"
        f"  End reason:    {ep.end_reason}\n"
        f"  Sub-goal reached at steps: "
        + (", ".join(str(s) for s in sub_goal_steps) if sub_goal_steps else "never")
        + peak_line
        + "\n\nTrajectory excerpt:\n" + "\n".join(excerpt_lines)
    )


# ── RL agent tool ─────────────────────────────────────────────────────────────

# Cache trained agents keyed by an agent_id so they can be reused across calls.
_trained_agents: dict[str, Any] = {}


class _ShapedEnv:
    """
    Thin wrapper around an RLIP environment that injects a sub-goal similarity
    bonus into every ``step()`` return value.

    The cosine similarity between the current observation's language description
    and **each** of the *sub_goal_languages* is computed at every step.  When
    the maximum similarity across the set meets or exceeds *threshold* a bonus
    of *bonus* is added to the reward.  Supporting multiple sub-goal languages
    means that any near-identical state to the primary match also triggers the
    shaped reward.

    All other attributes (``reset``, ``close``, ``action_space``, …) are
    forwarded to the wrapped environment unchanged.
    """

    def __init__(
        self,
        env: Any,
        sub_goal_language: str,
        bonus: float,
        threshold: float,
        translator: Any,
        env_id: str,
        sub_goal_languages: list[str] | None = None,
    ) -> None:
        self._env = env
        self._bonus = bonus
        self._threshold = threshold
        self._env_id = env_id
        # Full set of sub-goal language descriptions (primary + extras),
        # de-duplicated while preserving order.
        seen_sg: set[str] = set()
        self._all_sub_goal_languages: list[str] = []
        for lg in [sub_goal_language] + (sub_goal_languages or []):
            if lg not in seen_sg:
                seen_sg.add(lg)
                self._all_sub_goal_languages.append(lg)
        # Resolve language translator
        from ..language_translation import get_translator  # noqa: PLC0415
        from ..language_translation.base import LanguageTranslator  # noqa: PLC0415
        if isinstance(translator, LanguageTranslator):
            self._translator = translator
        else:
            self._translator = get_translator(env_id)
        # Encoder fitted lazily on first step
        self._encoder: Any = None
        self._sub_goal_vecs: list[Any] = []

    def _ensure_encoder(self) -> None:
        if self._encoder is not None:
            return
        from ..instruction_following import TextEncoder  # noqa: PLC0415
        enc = TextEncoder()
        enc.fit(self._all_sub_goal_languages)
        self._encoder = enc
        self._sub_goal_vecs = [enc.encode(lg) for lg in self._all_sub_goal_languages]

    def reset(self, seed: Any = None, options: Any = None) -> Any:
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        # Compute max similarity across all sub-goal descriptions and inject bonus.
        try:
            self._ensure_encoder()
            obs = result.observation if hasattr(result, "observation") else result.get("observation")
            if self._translator and obs is not None:
                lang = self._translator.translate(obs)
                obs_vec = self._encoder.encode(lang)
                sim = max(
                    float(self._encoder.cosine_similarity(obs_vec, sg_vec))
                    for sg_vec in self._sub_goal_vecs
                )
                if sim >= self._threshold:
                    if hasattr(result, "reward"):
                        object.__setattr__(result, "reward", result.reward + self._bonus)
                    elif isinstance(result, dict):
                        result = dict(result)
                        result["reward"] = result.get("reward", 0.0) + self._bonus
        except Exception:
            pass  # never crash the training loop over shaping
        return result

    def close(self) -> None:
        self._env.close()

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def env_id(self) -> str:
        return self._env_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class _LangStateEnv:
    """
    Environment wrapper that replaces raw observations with their
    natural-language translations at every ``reset()`` and ``step()``.

    When the translator returns an empty string or raises an exception the
    raw observation is returned unchanged as a safe fallback, so training
    always continues.

    Stack order when used together with ``_ShapedEnv``::

        _LangStateEnv(
            _ShapedEnv(base_env, ...)   ← injects reward bonus using raw obs
        )                                ← agent then sees language string obs

    The ``action_space``, ``env_id``, ``close``, and all other attributes
    are forwarded transparently to the wrapped environment.
    """

    def __init__(self, env: Any, translator: Any, env_id: str) -> None:
        self._env = env
        self._env_id = env_id
        from ..language_translation import get_translator  # noqa: PLC0415
        from ..language_translation.base import LanguageTranslator  # noqa: PLC0415
        if isinstance(translator, LanguageTranslator):
            self._translator: Any = translator
        else:
            self._translator = get_translator(env_id)

    def _translate(self, obs: Any) -> Any:
        """Return the language description of *obs*, or *obs* on failure."""
        if self._translator is None:
            return obs
        try:
            lang = self._translator.translate(obs)
            return lang if lang else obs
        except Exception:
            return obs

    def _apply_to_result(self, result: Any, key: str, translated: Any) -> Any:
        """Replace *key* in a Pydantic model or dict result."""
        if hasattr(result, key):
            object.__setattr__(result, key, translated)
        elif isinstance(result, dict):
            result = dict(result)
            result[key] = translated
        return result

    def reset(self, seed: Any = None, options: Any = None) -> Any:
        result = self._env.reset(seed=seed, options=options)
        if hasattr(result, "observation"):
            translated = self._translate(result.observation)
            result = self._apply_to_result(result, "observation", translated)
        elif isinstance(result, dict) and "observation" in result:
            translated = self._translate(result["observation"])
            result = dict(result)
            result["observation"] = translated
        else:
            # The result itself is the raw observation
            result = self._translate(result)
        return result

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        if hasattr(result, "observation"):
            translated = self._translate(result.observation)
            result = self._apply_to_result(result, "observation", translated)
        elif isinstance(result, dict) and "observation" in result:
            translated = self._translate(result["observation"])
            result = dict(result)
            result["observation"] = translated
        return result

    def close(self) -> None:
        self._env.close()

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def env_id(self) -> str:
        return self._env_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


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
        "agent": agent,
        "env_id": env_id,
        "agent_type": agent_type,
        "best_episode_history": result.best_episode_history,
        "use_language_state": use_language_state,
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


# ── Environment builder tools ─────────────────────────────────────────────────

# Sampled states held between rl_sample_states_for_translation and
# rl_set_translator_code so Claude doesn't need to pass them back.
_sampled_states: dict[str, list[Any]] = {}          # env_id → [(obs, action_history)]
# Active custom translators registered in this session.
_custom_translators: dict[str, Any] = {}            # env_id → LanguageTranslator


@mcp.tool()
def rl_build_environment(
    env_id: str,
    gym_env_id: str,
    description: str = "",
    tags: str = "",
    namespace: str = "custom",
    max_episode_steps: int = 0,
    translator_name: str = "",
) -> str:
    """
    Wrap a Gymnasium environment with custom metadata, cache it locally,
    and register it so it is immediately available in this session.

    The environment is saved to ``~/.rlip/envs/<env_id>/`` and written to
    ``~/.rlip/catalog.json`` so it is reloaded automatically on restart via
    rl_load_cached_environments().

    Parameters
    ----------
    env_id:
        The identifier to register this environment under (e.g.
        "FrozenLake-Custom-v0").  Can differ from gym_env_id.
    gym_env_id:
        The Gymnasium environment ID to wrap (e.g. "FrozenLake-v1").
    description:
        Human-readable description shown in rl_list_environments().
    tags:
        Comma-separated tags (e.g. "grid,discrete,custom").
    namespace:
        Registry namespace (default "custom").
    max_episode_steps:
        Hard episode step limit.  0 = use the Gymnasium default.
    translator_name:
        Optional name of a registered translator to attach (e.g.
        "Sailing-v0").  Leave blank to add a translator later with
        rl_set_translator_code().

    Returns
    -------
    Confirmation plus the local cache path.
    """
    from ..environments.builder import EnvironmentBuilder  # noqa: PLC0415

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    builder = (
        EnvironmentBuilder(env_id)
        .from_gymnasium(gym_env_id)
        .with_metadata(
            description=description,
            tags=tag_list,
            namespace=namespace,
            max_episode_steps=max_episode_steps or None,
        )
    )

    if translator_name:
        try:
            builder = builder.with_translator(translator_name)
        except ValueError as exc:
            return (
                f"Translator '{translator_name}' not found: {exc}\n"
                "Building environment without a translator.  You can add one "
                "later with rl_set_translator_code()."
            )

    try:
        built = builder.build()
    except Exception as exc:
        return f"Failed to build environment '{env_id}': {exc}"

    translator_line = (
        f"  Translator: {type(built.translator).__name__}"
        if built.translator
        else "  Translator: none  (use rl_set_translator_code() to add one)"
    )

    return (
        f"Environment '{env_id}' built and registered.\n\n"
        f"  Wraps:            {gym_env_id}\n"
        f"  Description:      {description or '(none)'}\n"
        f"  Tags:             {', '.join(tag_list) or '(none)'}\n"
        f"  Namespace:        {namespace}\n"
        f"{translator_line}\n"
        f"  Cache path:       {built.cache_path}\n\n"
        f"Use rl_create(env_id='{env_id}') to create an instance, or\n"
        f"rl_sample_states_for_translation(env_id='{env_id}') to start "
        f"building a language translator."
    )


@mcp.tool()
def rl_load_cached_environments() -> str:
    """
    Load all custom environments previously built with rl_build_environment()
    from ``~/.rlip/envs/`` and register them into this session.

    Call this once at the start of a session to restore environments that were
    created in a previous session.  Already-registered environments are
    re-registered without error (the latest cached version takes precedence).

    Returns
    -------
    A list of loaded environment IDs and their translator status.
    """
    from ..environments.builder import load_cached_environments  # noqa: PLC0415

    try:
        loaded = load_cached_environments()
    except Exception as exc:
        return f"Failed to load cached environments: {exc}"

    if not loaded:
        return (
            "No cached environments found in ~/.rlip/envs/.\n"
            "Use rl_build_environment() to create and cache a new environment."
        )

    # Also refresh in-process translator cache
    for built in loaded:
        if built.translator is not None:
            _custom_translators[built.spec.env_id] = built.translator

    lines = [f"Loaded {len(loaded)} cached environment(s):\n"]
    for built in loaded:
        has_t = "yes" if built.translator else "no"
        lines.append(
            f"  • {built.spec.env_id:40s}  translator={has_t}  "
            f"namespace={built.spec.namespace}"
        )
    return "\n".join(lines)


@mcp.tool()
def rl_list_cached_environments() -> str:
    """
    List all custom environments stored in the local cache (``~/.rlip/envs/``).

    Does not register them — call rl_load_cached_environments() to register.

    Returns summary metadata for each cached environment.
    """
    from ..environments.builder import _DEFAULT_CACHE_DIR  # noqa: PLC0415

    root = _DEFAULT_CACHE_DIR
    if not root.exists():
        return (
            "No cached environments found (~/.rlip/envs/ does not exist).\n"
            "Use rl_build_environment() to create your first custom environment."
        )

    entries: list[dict] = []
    for env_dir in sorted(root.iterdir()):
        spec_path = env_dir / "spec.json"
        if not spec_path.exists():
            continue
        try:
            entries.append(json.loads(spec_path.read_text()))
        except Exception:
            continue

    if not entries:
        return "No cached environments found."

    lines = [f"Cached environments ({len(entries)}):\n"]
    for spec in entries:
        has_t = "yes" if spec.get("language_translation") else "no"
        lines.append(
            f"  • {spec['env_id']}\n"
            f"    {spec.get('description', '(no description)')}\n"
            f"    tags={spec.get('tags', [])}  namespace={spec.get('namespace', '')}  "
            f"translator={has_t}\n"
            f"    created: {spec.get('created_at', '?')[:19]}"
        )
    return "\n".join(lines)


@mcp.tool()
def rl_sample_states_for_translation(
    env_id: str,
    n_samples: int = 20,
    seed: Optional[int] = None,
) -> str:
    """
    Sample unique observed states from an environment to help you write a
    language translator.

    Runs random exploration and collects up to *n_samples* distinct
    observations.  The sampled states are held in memory so you can pass
    ``rl_set_translator_code()`` immediately afterward without repeating them.

    Workflow
    --------
    1. Call this tool to see what the raw state values look like.
    2. Write a Python ``translate(state, ...)`` function that maps each state
       format to a natural-language description.
    3. Call ``rl_set_translator_code(env_id=..., python_code=...)`` to install
       and test it.

    Parameters
    ----------
    env_id:
        A registered RLIP environment ID.
    n_samples:
        Number of unique states to collect (default 20, max 50).
    seed:
        Optional seed for reproducible sampling.
    """
    from ..environments.registry import registry as _env_registry  # noqa: PLC0415
    from ..language_translation.generator import _sample_states  # noqa: PLC0415

    n_samples = min(n_samples, 50)

    try:
        factory = _env_registry.get(env_id)
        env = factory.create()
    except Exception as exc:
        return f"Could not create environment '{env_id}': {exc}"

    try:
        samples = _sample_states(env, n_samples=n_samples, seed=seed)
    except Exception as exc:
        return f"State sampling failed: {exc}"
    finally:
        try:
            env.close()
        except Exception:
            pass

    _sampled_states[env_id] = samples

    lines = [
        f"Sampled {len(samples)} unique states from '{env_id}'.\n",
        "Write a translate() function for these states, then call\n"
        f"rl_set_translator_code(env_id='{env_id}', python_code='...').\n",
        "Sampled states:",
    ]
    for i, (obs, ah) in enumerate(samples, 1):
        last = f"  (last action: {ah[-1]})" if ah else ""
        lines.append(f"  STATE {i:2d}: {repr(obs)}{last}")

    lines += [
        "",
        f"Example translate() function skeleton for '{env_id}':",
        "  def translate(state, *, legal_moves=None, action_history=None):",
        "      # Parse state and return a description string",
        "      # Return \"\" for states you cannot handle",
        f"      return f\"State: {{state}}\"",
    ]
    return "\n".join(lines)


@mcp.tool()
def rl_set_translator_code(
    env_id: str,
    python_code: str,
    save: bool = True,
) -> str:
    """
    Install a Python language translator function for an environment.

    Write a ``translate(state, *, legal_moves=None, action_history=None)``
    function and pass it here.  The translator is compiled and registered
    immediately so all subsequent protocol calls that use language translation
    (``translate=True``) pick it up automatically.

    Parameters
    ----------
    env_id:
        Environment to attach the translator to (must be registered with RLIP).
    python_code:
        Complete Python source of a ``translate`` function.  Example::

            def translate(state, *, legal_moves=None, action_history=None):
                parts = str(state).split("_")
                if len(parts) == 2:
                    return f"Position {parts[0]}, angle {parts[1]}"
                return ""

        Rules:
        - Must define a function named ``translate``.
        - Return a non-empty string for known states.
        - Return ``""`` (or raise) for states the function cannot handle —
          the system will fall back gracefully.
        - Do not import external packages; only Python built-ins are safe.
    save:
        If True (default), persist the translator to
        ``~/.rlip/envs/<env_id>/translator.py`` and update ``spec.json``
        so it is reloaded automatically by ``rl_load_cached_environments()``.

    Returns
    -------
    Confirmation plus test translations against previously sampled states
    (if rl_sample_states_for_translation was called for this env_id).
    """
    import datetime as _dt  # noqa: PLC0415

    from ..language_translation.generator import (  # noqa: PLC0415
        GeneratedTranslator,
        _strip_markdown_fences,
        _to_class_name,
    )
    from ..language_translation import TRANSLATORS  # noqa: PLC0415

    code = _strip_markdown_fences(python_code)

    def _no_llm(prompt: str) -> str:
        return ""

    try:
        gt = GeneratedTranslator(
            llm_fn=_no_llm,
            env_id=env_id,
            rule_code=code,
            env_context="",
            refine_threshold=99_999,  # disable auto-refine for hand-written rules
        )
    except ValueError as exc:
        return (
            f"Could not compile translator code for '{env_id}': {exc}\n\n"
            "Ensure the code defines a function with this exact signature:\n"
            "  def translate(state, *, legal_moves=None, action_history=None): ..."
        )

    _custom_translators[env_id] = gt
    TRANSLATORS[env_id] = gt

    cache_msg = ""
    if save:
        cache_dir = Path.home() / ".rlip" / "envs" / env_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        class_name = _to_class_name(env_id)
        module_path = cache_dir / "translator.py"
        gt.save_code(module_path, class_name=class_name)

        spec_path = cache_dir / "spec.json"
        if spec_path.exists():
            try:
                spec = json.loads(spec_path.read_text())
                spec["language_translation"] = {
                    "type": "generated",
                    "module_path": "translator.py",
                    "class_name": class_name,
                }
                spec["updated_at"] = _dt.datetime.now().isoformat()
                spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            except Exception as exc:
                cache_msg = f"\n(spec.json update failed: {exc})"
        cache_msg = f"\nSaved to: {module_path}" + cache_msg

    # Test against previously sampled states (up to 8)
    test_lines: list[str] = []
    if env_id in _sampled_states:
        test_lines.append("\nTest translations on sampled states:")
        for obs, _ah in _sampled_states[env_id][:8]:
            try:
                desc = gt.translate(obs)
                result_str = repr(desc) if desc else '""  ← fallback will apply'
            except Exception as exc:
                result_str = f"ERROR: {exc}"
            test_lines.append(f"  {str(repr(obs)):42s} → {result_str}")

    return (
        f"Translator installed for '{env_id}'.\n"
        f"  Compiled rule function: yes\n"
        f"  Auto-registered in TRANSLATORS: yes"
        + cache_msg
        + ("\n" + "\n".join(test_lines) if test_lines else "")
        + f"\n\nUse translate=True in any protocol call, or test with:\n"
        f"  rl_translate_state(env_id='{env_id}', state='...')"
    )


@mcp.tool()
def rl_translate_state(
    env_id: str,
    state: str,
) -> str:
    """
    Translate a raw environment observation to its natural-language description.

    Useful for inspecting translation quality or debugging a translator before
    using it in instruction-matching or protocol calls.

    Parameters
    ----------
    env_id:
        A registered RLIP environment ID.
    state:
        The raw state value encoded as a JSON string.
        - String observations: ``'"0.0300_0.2"'``
        - Integer observations: ``"3"``
        - List/array observations: ``"[1, 0, 2]"``

    Returns
    -------
    The natural-language description produced by the registered translator.
    """
    from ..language_translation import get_translator  # noqa: PLC0415

    translator = _custom_translators.get(env_id) or get_translator(env_id)
    if translator is None:
        return (
            f"No translator registered for '{env_id}'.\n"
            "Use rl_set_translator_code() to install one, or\n"
            "rl_sample_states_for_translation() to start the workflow."
        )

    try:
        obs = json.loads(state)
    except json.JSONDecodeError:
        obs = state  # treat as a raw string

    try:
        description = translator.translate(obs)
    except Exception as exc:
        return f"Translation raised an exception: {exc}"

    if not description:
        return (
            f"Translator returned empty string for state {state!r}.\n"
            "The state may be out of range or in an unrecognised format.\n"
            "Update rl_set_translator_code() to handle this state."
        )

    return f"Translation for {env_id} | state={state!r}\n  → {description}"


# ── Resources ────────────────────────────────────────────────────────────────

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


# ── Renders directory (persists locally, easy for Claude to reference) ────────

_RENDERS_DIR = Path.home() / ".rlip" / "renders"


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

    Runs several episodes, picks the best one, replays it with rendering
    enabled, and saves the result as a ``.gif`` file.  The GIF is returned
    as an inline data URL so Claude can display it directly, and also saved
    to ``~/.rlip/renders/`` for local access.

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
    import time
    from datetime import datetime

    from ..interaction_protocols import (
        MultiEpisodeProtocol,
        RandomEpisodeProtocol,
        GreedyEpisodeProtocol,
    )
    from ..policy_rendering import render_optimal_policy

    # ── Resolve environment factory ───────────────────────────────────────────
    if env_id not in _registry:
        return f"Unknown environment '{env_id}'. Use rl_list_environments() to see available envs."

    factory = _registry.get(env_id)

    # ── Determine GIF output path ─────────────────────────────────────────────
    _RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = env_id.replace("/", "_").replace("-", "_").replace(" ", "_")
    _stored = _trained_agents.get(agent_id) if agent_id else None
    agent_type_label = _stored.get("agent_type", "unknown") if _stored else "random"
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = _RENDERS_DIR / f"{safe_id}_{agent_type_label}_{n_episodes}ep_{_ts}.gif"

    # ── Collect episodes ──────────────────────────────────────────────────────
    train_env = factory.create(render_mode=None)
    try:
        if agent_id and agent_id in _trained_agents:
            # Use stored best-training-episode history directly.
            # This avoids re-running the greedy policy, which may cycle for
            # agents like TabularQ whose Q-values didn't fully propagate.
            stored = _trained_agents[agent_id]
            training_history = stored.get("best_episode_history", [])
            if training_history:
                from ..policy_rendering import PolicyRenderer  # noqa: PLC0415
                from ..policy_rendering import save_gif, PolicyRenderResult  # noqa: PLC0415
                from ..policy_rendering import _hashable_obs  # noqa: PLC0415

                policy = {_hashable_obs(obs): act for obs, act in training_history}
                render_env = factory.create(render_mode="rgb_array")
                renderer = PolicyRenderer(env=render_env, policy=policy)
                frames = renderer.run(max_steps=max_steps, seed=seed)

                n_gif_frames = save_gif(frames, gif_path, fps=fps, annotate=True)
                if n_gif_frames == 0:
                    return (
                        f"Policy replay completed but no frames were captured.\n"
                        f"Environment '{env_id}' may not support rgb_array rendering."
                    )

                agent_type = stored.get("agent_type", "unknown")
                best_reward = max((r for _, r in [(0, 0)] + [(0, 0)]), default=0)
                # Compute best reward from stored history rewards isn't available here;
                # report frame count instead.
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


# ── Entry-point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Entry-point called by the ``rlip-mcp`` console script."""
    logging.basicConfig(level=logging.WARNING)
    # Auto-register all Gymnasium environments that are installed
    if _in_process:
        try:
            count = _registry.auto_register_gymnasium()
            log.info("Auto-registered %d Gymnasium environments.", count)
        except Exception as exc:
            log.warning("auto_register_gymnasium failed: %s", exc)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
