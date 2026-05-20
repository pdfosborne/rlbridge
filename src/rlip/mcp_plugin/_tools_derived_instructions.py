"""
Derived-instruction MCP tools for the RLIP plugin.

Tools:
  rl_train_and_derive_instructions  – train an RL agent while tracking
      language state visits, then automatically generate the top instruction
      candidates most correlated with success and cache them.
  rl_apply_derived_instruction      – take a derived instruction from the
      cache and set it up as a sub-goal for rl_instruction_run_episode /
      rl_train_agent (returns a match_id).

Instruction cache inspection tools (rl_list_cached_instructions,
rl_clear_instruction_cache) live in _tools_instruction_plan.py.
"""

from __future__ import annotations

import asyncio
import re
import sys
import uuid
from typing import Any, Optional

from mcp.server.fastmcp import Context

from ._prompts import decompose_instruction_simple_prompt
from ._state import (
    _custom_translators,
    _in_process,
    _instruction_protocols,
    _trained_agents,
    _training_jobs,
    log,
    mcp,
)


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _decompose_instruction_with_llm(
    ctx: Context,
    instruction: str,
    env_id: str,
    observed_langs: list[str],
) -> list[str]:
    """Decompose one instruction into ordered, distinct, observable sub-steps."""
    import mcp.types as _t

    n_sample = min(20, len(observed_langs))
    step = max(1, len(observed_langs) // n_sample)
    sample = observed_langs[::step][:n_sample]
    lang_block = "\n".join(f"  - {lg}" for lg in sample)
    if len(observed_langs) > n_sample:
        lang_block += f"\n  ... ({len(observed_langs) - n_sample} more not shown)"

    prompt = decompose_instruction_simple_prompt(env_id, instruction, lang_block)

    try:
        result = await ctx.session.create_message(
            messages=[_t.SamplingMessage(
                role="user",
                content=_t.TextContent(type="text", text=prompt),
            )],
            max_tokens=256,
        )
    except Exception:
        return []

    content = result.content
    if hasattr(content, "text"):
        raw = content.text
    elif isinstance(content, list) and content:
        raw = getattr(content[0], "text", "") or ""
    else:
        raw = str(content)

    out: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^[\d]+[.)]\s*|^[-*]\s*", "", line).strip(" .")
        if cleaned:
            out.append(cleaned)
    return out


def _fallback_decompose_instruction(instruction: str) -> list[str]:
    """Deterministic decomposition when LLM output is unavailable."""
    # Only split on explicit sequential connectors — never on commas, which
    # would fragment instruction text into meaningless phrase fragments.
    parts = [
        p.strip(" .")
        for p in re.split(r"\bthen\b|\band then\b", instruction, flags=re.IGNORECASE)
        if p.strip(" .")
    ]
    uniq: list[str] = []
    seen: set[str] = set()
    for p in parts:
        k = re.sub(r"\s+", " ", p).lower()
        if len(k) < 3 or k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    if len(uniq) >= 2:
        return uniq[:5]
    # Single instruction — return it as-is rather than inventing sub-steps.
    return [instruction.strip().rstrip(".")]


def _normalize_steps(instruction: str, llm_steps: list[str]) -> list[str]:
    raw = llm_steps if llm_steps else _fallback_decompose_instruction(instruction)
    out: list[str] = []
    seen: set[str] = set()
    for step in raw:
        cleaned = re.sub(r"\s+", " ", step).strip(" .")
        key = cleaned.lower()
        if not cleaned or len(cleaned) < 3 or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= 5:
            break
    if len(out) < 2:
        return _fallback_decompose_instruction(instruction)
    return out

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
def rl_train_and_derive_instructions(
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

    job_id = uuid.uuid4().hex[:12]
    _training_jobs[job_id] = {
        "status": "running",
        "agent_id": None,   # filled in by _job once training completes
        "env_id": env_id,
        "n_episodes": n_episodes,
        "progress_env": progress_env,
    }

    def _job() -> None:
        try:
            train_result = agent.train(progress_env, n_episodes=n_episodes, max_steps=max_steps)
        except Exception as exc:
            _training_jobs[job_id]["status"] = "failed"
            _training_jobs[job_id]["result"] = f"Training failed: {exc}"
            try:
                progress_env._bar.close()
            except Exception:
                pass
            return

        try:
            progress_env._bar.close()
        except Exception:
            pass

        # Derive instructions from the language visit log.
        derived = derive_instructions_from_training(
            progress_env.language_wrapper,
            top_k=top_k,
            min_episode_visits=min_episode_visits,
        )

        # ── Instruction plan database: register derived entries ───────────────
        try:
            import math as _math
            from ..instruction_plan_db import get_plan_database
            from ._state import _env_plan_db_path
            _wrapper = progress_env.language_wrapper
            _plan_db = get_plan_database(env_id, plan_path=str(_env_plan_db_path(env_id)))
            for _entry in derived:
                _lang = _entry.instruction
                _ep_set = _wrapper._lang_episodes.get(_lang, set())
                _n_eps = len(_ep_set)
                _n_success = len(_wrapper._lang_success_episodes.get(_lang, set()))
                _csr = _n_success / _n_eps if _n_eps > 0 else 0.0
                _score = _csr * _math.log2(1.0 + _n_eps)
                _plan_db.add_derived(
                    instruction=_lang,
                    derived_score=_score,
                    derived_csr=_csr,
                    similarity=1.0,
                )
        except Exception:
            pass

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
        _training_jobs[job_id]["agent_id"] = agent_id

        if not derived:
            result_text = (
                f"Training complete ({n_episodes} episodes) but no instruction "
                f"candidates could be derived (too few successful episodes or "
                f"min_episode_visits={min_episode_visits} not satisfied).\n\n"
                f"Agent saved as agent_id='{agent_id}'.\n"
                f"Mean reward: {train_result.mean_reward:.4f}  "
                f"Best reward: {train_result.best_reward:.4f}"
            )
        else:
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
            result_text = "\n".join(lines)

        _training_jobs[job_id]["result"] = result_text
        _training_jobs[job_id]["status"] = "done"

    import threading as _threading
    _threading.Thread(target=_job, daemon=True).start()

    return (
        f"Training started — {agent_type} on {env_id}\n\n"
        f"  Job ID:    {job_id}\n"
        f"  Episodes:  {n_episodes}  max_steps={max_steps}\n\n"
        f"Training runs in the background with no time limits.\n"
        f"Call rl_get_training_result(job_id='{job_id}') to check progress and get the result.\n"
        f"The agent_id and derived instructions will be available when training completes."
    )

# rl_list_cached_instructions and rl_clear_instruction_cache live in
# _tools_instruction_plan.py.


@mcp.tool()
async def rl_apply_derived_instruction(
    ctx: Context,
    env_id: str,
    instruction: str,
    sub_goal_bonus: float = 0.0,
    sub_goal_threshold: float = 0.5,
    sub_goal_repeatable: bool = False,
    max_steps: int = 200,
    seed: Optional[int] = None,
) -> str:
    """
    Look up a cached instruction (from rl_list_cached_instructions), decompose
    it into ordered steps, and convert it into a live sequential protocol,
    returning a
    match_id that can be passed to rl_instruction_run_episode() or
    rl_train_agent(match_id=...).

    This is the bridge between automatically derived instructions and the
    sequential instruction-following workflow. The obs cache is already
    populated (from training), so no re-exploration is needed.

    Parameters
    ----------
    env_id:
        The environment the instruction belongs to.
    instruction:
        The exact instruction string from rl_list_cached_instructions().
    sub_goal_bonus:
        Bonus reward when each sequential stage is reached. Set to
        0.0 (default) to auto-scale.
    sub_goal_threshold:
        Cosine similarity threshold (0–1) to award the bonus.
    sub_goal_repeatable:
        Kept for API compatibility. Ignored in sequential mode.
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
        build_sequential_instruction_following_protocol,
        obs_cache_langs,
    )
    from ..environments.registry import registry as _env_registry

    env_entries = _INSTRUCTION_CACHE.get(env_id, {})
    del sub_goal_repeatable
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

    observed_langs = obs_cache_langs(env_id)
    llm_steps = await _decompose_instruction_with_llm(
        ctx, instruction, env_id, observed_langs
    )
    steps = _normalize_steps(instruction, llm_steps)

    # build_sequential_instruction_following_protocol skips re-exploration
    # when the observation cache is already populated.
    try:
        protocol = build_sequential_instruction_following_protocol(
            steps,
            env,
            sub_goal_bonus=None if sub_goal_bonus == 0.0 else sub_goal_bonus,
            sub_goal_threshold=sub_goal_threshold,
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
        "is_sequential": True,
        "instructions": steps,
        "decomposition": steps,
    }

    entry = env_entries[instruction]
    sr = f"{entry.success_rate:.1%}" if entry.episodes_run > 0 else "n/a"
    effective_bonus = protocol.sub_goal_bonus

    return (
        f"Instruction applied for '{env_id}':\n\n"
        f"  Instruction:    {instruction!r}\n"
        f"  Similarity:     {entry.match.similarity_score:.4f}\n"
        f"  Sequential steps: {len(steps)}\n"
        + "\n".join(f"    {i+1}. {s}" for i, s in enumerate(steps))
        + "\n"
        f"  Sub-goal bonus: {effective_bonus:.6g} (auto-scaled)\n"
        f"  Training stats: {entry.episodes_run} episode(s)  "
        f"success_rate={sr}\n"
        f"  Match ID:       {match_id}\n\n"
        f"Use rl_instruction_run_episode(match_id='{match_id}') to run a "
        f"sequentially shaped episode.\n"
        f"Or pass match_id='{match_id}' to rl_train_agent() for sequential shaped training."
    )
