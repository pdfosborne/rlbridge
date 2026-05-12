"""
Derived-instruction MCP tools for the RLIP plugin.

Tools:
  rl_train_and_derive_instructions  – train an RL agent while tracking
      language state visits, then automatically generate the top instruction
      candidates most correlated with success and cache them.
  rl_list_cached_instructions       – inspect the instruction cache for an
      environment (with per-instruction success-rate statistics).
  rl_clear_instruction_cache        – discard cached instructions (and the
      state→instruction map) for one or all environments.
  rl_apply_derived_instruction      – take a derived instruction from the
      cache and set it up as a sub-goal for rl_instruction_run_episode /
      rl_train_agent (returns a match_id).
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from typing import Any, Optional

from mcp.server.fastmcp import Context

from ._state import (
    _RENDERS_DIR,
    _custom_translators,
    _in_process,
    _instruction_protocols,
    _trained_agents,
    log,
    mcp,
)


# ── Internal helpers ──────────────────────────────────────────────────────────

class _TrackingProgressEnv:
    """
    Combines LanguageTrackingWrapper with a tqdm progress bar so the MCP
    client sees training progress while state-visit statistics are collected.
    """

    def __init__(
        self,
        env: Any,
        n_episodes: int,
        agent_type: str,
        env_id: str,
        translator: Any,
        success_fn: Any,
    ) -> None:
        import tqdm
        from ..instruction_following import LanguageTrackingWrapper

        self._wrapper = LanguageTrackingWrapper(
            env, translator=translator, success_fn=success_fn
        )
        self._n_episodes = n_episodes
        self._best_reward = float("-inf")
        self._ep_reward = 0.0
        self._reset_calls = 0
        self._bar = tqdm.tqdm(
            total=n_episodes,
            desc=f"Training {agent_type} on {env_id} (tracking)",
            unit="ep",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
        )

    # ---- env interface -------------------------------------------------------

    def reset(self, seed: Any = None, options: Any = None) -> Any:
        if self._reset_calls > 0:
            if self._ep_reward > self._best_reward:
                self._best_reward = self._ep_reward
            self._bar.set_postfix(
                last=f"{self._ep_reward:.2f}",
                best=f"{self._best_reward:.2f}",
            )
            n = min(1, self._n_episodes - self._bar.n)
            if n > 0:
                self._bar.update(n)
            self._ep_reward = 0.0
        self._reset_calls += 1
        return self._wrapper.reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:
        result = self._wrapper.step(action)
        try:
            r = result.reward if hasattr(result, "reward") else result.get("reward", 0.0)
            self._ep_reward += float(r)
        except Exception:
            pass
        return result

    def close(self) -> None:
        self._bar.close()
        self._wrapper.close()

    @property
    def action_space(self) -> Any:
        return self._wrapper.action_space

    @property
    def env_id(self) -> str:
        return self._wrapper.env_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapper, name)

    # expose the inner wrapper for derive_instructions_from_training
    @property
    def language_wrapper(self):  # type: ignore[return]
        return self._wrapper


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def rl_train_and_derive_instructions(
    ctx: Context,
    env_id: str,
    agent_type: str = "tabular_q",
    n_episodes: int = 300,
    max_steps: int = 200,
    seed: Optional[int] = None,
    top_k: int = 5,
    min_episode_visits: int = 2,
    # Tabular Q hyper-parameters
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    # DQN / PPO shared
    hidden_size: int = 64,
    lr: float = 1e-3,
) -> str:
    """
    Train an RL agent on an environment while tracking which language-described
    states are visited during successful episodes.  After training, the top-k
    states most correlated with success are saved as cached instructions
    that can immediately be used with rl_apply_derived_instruction() or
    rl_instruction_run_episode().

    How it works
    ------------
    The environment is wrapped with a LanguageTrackingWrapper that translates
    each observation to its natural-language description at every step.  After
    *n_episodes* training episodes the wrapper holds:

      - How many distinct training episodes visited each described state.
      - How many of those episodes were *successful* (ended with
        ``terminated=True`` by default).

    Each state is scored by::

        score = success_rate × log₂(1 + episode_visits)

    The top-*k* states are registered in the instruction cache so they can be
    recalled instantly by rl_list_cached_instructions() and turned into MCP
    sub-goal protocols by rl_apply_derived_instruction().

    Parameters
    ----------
    env_id:
        A registered RLIP environment ID, e.g. ``"Sailing-v0"``.
    agent_type:
        One of ``"tabular_q"``, ``"dqn"``, or ``"ppo"``.
    n_episodes:
        Number of training episodes.
    max_steps:
        Step cap per episode.
    seed:
        Optional reproducibility seed.
    top_k:
        Maximum number of instruction candidates to derive (1–20).
    min_episode_visits:
        Ignore states visited in fewer than this many distinct episodes.
    alpha / gamma / epsilon / ...:
        Agent hyper-parameters (forwarded to the appropriate agent type).

    Returns
    -------
    A text summary listing the derived instructions with their success rates
    and an agent_id for the trained agent.
    """
    from ..environments.registry import registry as _env_registry
    from ..instruction_following import derive_instructions_from_training
    from ..language_translation import get_translator
    from ..rl_agents import DQNAgent, PPOAgent, TabularQAgent

    agent_type = agent_type.lower().strip()
    valid = ("tabular_q", "dqn", "ppo")
    if agent_type not in valid:
        return (
            f"Unknown agent type '{agent_type}'.  "
            f"Valid choices: {', '.join(valid)}."
        )

    try:
        factory = _env_registry.get(env_id)
        env = factory.create()
    except KeyError:
        return (
            f"Environment '{env_id}' is not registered.  "
            "Call rl_list_environments() to see available environments."
        )

    translator = _custom_translators.get(env_id) or get_translator(env_id)
    if translator is None:
        env.close()
        return (
            f"No language translator found for '{env_id}'.  "
            "Register one with rl_set_translator_code() first."
        )

    top_k = max(1, min(top_k, 20))

    progress_env = _TrackingProgressEnv(
        env,
        n_episodes=n_episodes,
        agent_type=agent_type,
        env_id=env_id,
        translator=translator,
        success_fn=None,
    )

    if agent_type == "tabular_q":
        agent = TabularQAgent(
            alpha=alpha, gamma=gamma,
            epsilon=epsilon, epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay, seed=seed,
        )
    elif agent_type == "dqn":
        agent = DQNAgent(
            hidden_size=hidden_size, lr=lr, gamma=gamma,
            epsilon=epsilon, epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay, seed=seed,
        )
    else:
        agent = PPOAgent(
            hidden_size=hidden_size, lr_actor=lr, gamma=gamma, seed=seed,
        )

    async def _poll() -> None:
        ep_approx = 0
        while True:
            await asyncio.sleep(0.5)
            ep_approx = progress_env._reset_calls
            await ctx.report_progress(ep_approx, n_episodes)

    poll_task = asyncio.create_task(_poll())
    try:
        train_result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: agent.train(progress_env, n_episodes=n_episodes, max_steps=max_steps),
        )
    except Exception as exc:
        poll_task.cancel()
        return f"Training failed: {exc}"
    finally:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass
        progress_env._bar.close()

    # Derive instructions from the language visit log.
    derived = derive_instructions_from_training(
        progress_env.language_wrapper,
        top_k=top_k,
        min_episode_visits=min_episode_visits,
    )

    # Store the trained agent for later use (rl_run_agent_episode, etc.)
    agent_id = uuid.uuid4().hex[:12]
    _trained_agents[agent_id] = {
        "agent":                agent,
        "agent_type":           agent_type,
        "env_id":               env_id,
        "train_result":         train_result,
        "use_language_state":   False,
        "best_episode_history": getattr(train_result, "best_episode_history", []),
        "derived_instructions": [e.instruction for e in derived],
    }

    if not derived:
        return (
            f"Training complete ({n_episodes} episodes) but no instruction "
            f"candidates could be derived (too few successful episodes or "
            f"min_episode_visits={min_episode_visits} not satisfied).\n\n"
            f"Agent saved as agent_id='{agent_id}'.\n"
            f"Mean reward: {train_result.mean_reward:.4f}  "
            f"Best reward: {train_result.best_reward:.4f}"
        )

    lines = [
        f"Derived {len(derived)} instruction candidate(s) for '{env_id}' "
        f"after {n_episodes} training episodes:\n"
    ]
    for i, entry in enumerate(derived, 1):
        sr = entry.success_rate
        lines.append(
            f"  {i}. success_rate={sr:.1%}  episodes_seen={entry.episodes_run}\n"
            f"     instruction: {entry.instruction!r}\n"
        )

    lines.append(
        f"\nAgent saved as agent_id='{agent_id}'.\n"
        f"Mean reward: {train_result.mean_reward:.4f}  "
        f"Best reward: {train_result.best_reward:.4f}\n\n"
        "Use rl_list_cached_instructions(env_id) to inspect the full cache.\n"
        "Use rl_apply_derived_instruction(env_id, instruction) to convert a "
        "derived instruction into a match_id for rl_instruction_run_episode()."
    )
    return "\n".join(lines)


@mcp.tool()
def rl_list_cached_instructions(env_id: str = "") -> str:
    """
    List all cached instruction entries for one environment (or all
    environments), showing each instruction's matched state, similarity
    score, and accumulated episode success statistics.

    Instructions are populated by:
      • rl_match_instruction() — one entry per explicit user instruction
      • rl_train_and_derive_instructions() — auto-derived entries

    Parameters
    ----------
    env_id:
        Restrict output to this environment.  When empty, all environments
        are shown.

    Returns
    -------
    A text table of cached instructions with success-rate statistics.
    """
    from ..instruction_following import list_instruction_cache

    cache = list_instruction_cache(env_id.strip() or None)
    if not cache:
        scope = f"'{env_id}'" if env_id.strip() else "any environment"
        return f"No cached instructions found for {scope}."

    lines: list[str] = []
    for eid, entries in sorted(cache.items()):
        lines.append(f"Environment: {eid}  ({len(entries)} instruction(s))\n")
        for instr, entry in entries.items():
            sr = f"{entry.success_rate:.1%}" if entry.episodes_run > 0 else "n/a"
            lines.append(
                f"  instruction:   {instr!r}\n"
                f"  matched state: {entry.match.matched_language!r}\n"
                f"  similarity:    {entry.match.similarity_score:.4f}\n"
                f"  episodes_run:  {entry.episodes_run}  "
                f"successes={entry.successes}  success_rate={sr}\n"
            )
        lines.append("")

    lines.append(
        "Use rl_apply_derived_instruction(env_id, instruction) to turn any of "
        "these into a match_id for rl_instruction_run_episode() or rl_train_agent()."
    )
    return "\n".join(lines)


@mcp.tool()
def rl_clear_instruction_cache(env_id: str = "") -> str:
    """
    Clear the instruction cache (and the state→instruction map) for one
    environment or all environments.

    Parameters
    ----------
    env_id:
        Clear only this environment.  When empty the entire cache is cleared.

    Returns
    -------
    Confirmation string.
    """
    from ..instruction_following import (
        clear_instruction_cache,
        list_instruction_cache,
    )

    target = env_id.strip() or None
    before = list_instruction_cache(target)
    total = sum(len(v) for v in before.values())
    clear_instruction_cache(target)

    scope = f"'{target}'" if target else "all environments"
    return (
        f"Instruction cache cleared for {scope}.\n"
        f"Removed {total} cached instruction(s)."
    )


@mcp.tool()
def rl_apply_derived_instruction(
    env_id: str,
    instruction: str,
    sub_goal_bonus: float = 0.0,
    sub_goal_threshold: float = 0.5,
    sub_goal_repeatable: bool = False,
    max_steps: int = 200,
    seed: Optional[int] = None,
) -> str:
    """
    Look up a cached instruction (from rl_list_cached_instructions) and
    convert it into a live InstructionFollowingProtocol, returning a
    match_id that can be passed to rl_instruction_run_episode() or
    rl_train_agent(match_id=...).

    This is the bridge between automatically derived instructions and the
    existing instruction-following workflow.  The obs cache is already
    populated (from training), so no re-exploration is needed.

    Parameters
    ----------
    env_id:
        The environment the instruction belongs to.
    instruction:
        The exact instruction string from rl_list_cached_instructions().
    sub_goal_bonus:\n        Bonus reward when the sub-goal similarity threshold is met.  Set to\n        0.0 (default) to auto-scale: ``max_reward / (100 \u00d7 n_sub_goals)``.\n        Pass an explicit positive value to override.
    sub_goal_threshold:
        Cosine similarity threshold (0–1) to award the bonus.
    sub_goal_repeatable:
        False (default) – bonus awarded once per episode.
        True – bonus awarded every step the threshold is met.
    max_steps:
        Episode step cap stored on the protocol.
    seed:
        Optional seed for subsequent episode resets.

    Returns
    -------
    A match_id string and a description of the sub-goal, or an error message
    if the instruction is not in the cache.
    """
    from ..instruction_following import (
        _INSTRUCTION_CACHE,
        _STATE_INSTRUCTIONS,
        build_instruction_following_protocol,
        list_instruction_cache,
    )
    from ..environments.registry import registry as _env_registry

    env_entries = _INSTRUCTION_CACHE.get(env_id, {})
    if instruction not in env_entries:
        available = list(env_entries.keys())
        hint = (
            f"\n  Available for '{env_id}': "
            + (", ".join(repr(k) for k in available[:5]) or "(none)")
            + ("..." if len(available) > 5 else "")
        ) if available else ""
        return (
            f"Instruction {instruction!r} not found in cache for '{env_id}'.{hint}\n"
            "Run rl_list_cached_instructions() to see what is cached, or "
            "rl_train_and_derive_instructions() to generate new candidates."
        )

    try:
        factory = _env_registry.get(env_id)
        env = factory.create()
    except KeyError:
        return (
            f"Environment '{env_id}' is not registered.  "
            "Call rl_list_environments() to see available environments."
        )

    # build_instruction_following_protocol will skip re-exploration because
    # _OBS_CACHE is already populated from the training run.
    try:
        protocol = build_instruction_following_protocol(
            instruction,
            env,
            sub_goal_bonus=None if sub_goal_bonus == 0.0 else sub_goal_bonus,
            sub_goal_threshold=sub_goal_threshold,
            sub_goal_repeatable=sub_goal_repeatable,
            max_steps=max_steps,
            seed=seed,
        )
    except Exception as exc:
        env.close()
        return f"Failed to build instruction-following protocol: {exc}"

    match_id = uuid.uuid4().hex[:12]
    _instruction_protocols[match_id] = {
        "protocol": protocol,
        "env_id":   env_id,
        "env":      env,
    }

    entry = env_entries[instruction]
    sr = f"{entry.success_rate:.1%}" if entry.episodes_run > 0 else "n/a"
    effective_bonus = protocol.sub_goal_bonus  # already resolved by build_instruction...

    # Also list any other instructions that overlap with this state.
    state_map = _STATE_INSTRUCTIONS.get(env_id, {})
    matched_lang = protocol.sub_goal_language
    co_instrs = [i for i in state_map.get(matched_lang, []) if i != instruction]
    co_str = (
        "\n  Co-mapped instructions: " + ", ".join(repr(i) for i in co_instrs[:3])
        + ("..." if len(co_instrs) > 3 else "")
    ) if co_instrs else ""

    return (
        f"Instruction applied for '{env_id}':\n\n"
        f"  Instruction:    {instruction!r}\n"
        f"  Matched state:  {matched_lang!r}\n"
        f"  Similarity:     {entry.match.similarity_score:.4f}\n"
        f"  Sub-goal bonus: {effective_bonus:.6g} (auto-scaled)\n"
        f"  Training stats: {entry.episodes_run} episode(s)  "
        f"success_rate={sr}{co_str}\n"
        f"  Match ID:       {match_id}\n\n"
        f"Use rl_instruction_run_episode(match_id='{match_id}') to run a "
        f"sub-goal-shaped episode.\n"
        f"Or pass match_id='{match_id}' to rl_train_agent() for shaped training."
    )
