"""
LLM direct-play MCP tool for the rlbridge plugin.

Tool: rl_llm_play_episode

Lets the connected LLM act as the policy for one full episode - optionally
on both the raw environment and the language-translated form - and reports
whether it completed the episode successfully.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import mcp.types as _mcp_t
from mcp.server.fastmcp import Context

from ._state import _custom_translators, mcp


@mcp.tool()
async def rl_llm_play_episode(
    env_id: str,
    ctx: Context,
    max_steps: int = 200,
    seed: Optional[int] = None,
    modes: str = "raw,language",
) -> str:
    """
    Let the connected LLM act as the policy for one full episode and report
    whether it solved the task.

    The LLM is called once per step to choose an action.  This runs directly
    in the tool body (not a background thread) so the LLM can see each
    request as it happens and the result is returned synchronously.

    Parameters
    ----------
    env_id:
        Environment to play, e.g. "CartPole-v1".
    max_steps:
        Maximum steps per episode (per mode).
    seed:
        Optional seed for the environment reset.
    modes:
        Comma-separated list of play modes.  Supported values:
        ``raw``      - observations passed as-is (numeric / dict).
        ``language`` - observations translated to natural-language strings via
                       the registered translator for this environment.
        Default: ``"raw,language"`` (runs both).

    Returns
    -------
    Summary for each requested mode including steps taken, total reward,
    episode outcome (terminated / truncated / incomplete), and token usage.
    """
    from ..environments.registry import registry as _reg  # noqa: PLC0415
    from ..language_translation import get_translator as _get_trans  # noqa: PLC0415
    from ._env_wrappers import _LangStateEnv  # noqa: PLC0415
    from .._agent_base import _get as _ag_get  # noqa: PLC0415

    mode_list = [m.strip().lower() for m in modes.split(",") if m.strip()]
    if not mode_list:
        return "No valid modes specified.  Use 'raw', 'language', or 'raw,language'."

    try:
        factory = _reg.get(env_id)
    except KeyError:
        return (
            f"Environment '{env_id}' not found.  "
            "Use rl_list_environments() to see available environments."
        )

    translator = _custom_translators.get(env_id) or _get_trans(env_id)

    _SYS = (
        "You are a policy model for a reinforcement-learning environment. "
        "Choose exactly one valid action based on the current observation. "
        'Return only JSON with shape {"action": <value>} and no extra text.'
    )

    async def _run_mode(use_language: bool) -> dict[str, Any]:
        """Run one episode in the given mode; returns a result dict."""
        env = factory.create()
        try:
            if use_language and translator is not None:
                env = _LangStateEnv(env, translator=translator, env_id=env_id)

            reset_out = env.reset(seed=seed)
            obs = _ag_get(reset_out, "observation", reset_out)
            action_space = getattr(env, "action_space", None)

            if action_space is not None and hasattr(action_space, "n"):
                space_desc: Any = {"type": "Discrete", "n": int(action_space.n)}
            elif action_space is not None:
                space_desc = repr(action_space)
            else:
                space_desc = None

            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            total_tokens = 0
            t0 = time.time()

            for _ in range(max_steps):
                prompt = _SYS + "\n" + json.dumps(
                    {
                        "observation": obs,
                        "action_space": space_desc,
                        "instruction": (
                            'Choose the next action and return strict JSON: {"action": ...}'
                        ),
                    },
                    ensure_ascii=True,
                )

                msg = await ctx.session.create_message(
                    messages=[
                        _mcp_t.SamplingMessage(
                            role="user",
                            content=_mcp_t.TextContent(type="text", text=prompt),
                        )
                    ],
                    max_tokens=64,
                )
                raw = (
                    msg.content.text
                    if hasattr(msg.content, "text")
                    else str(msg.content)
                )
                total_tokens += len(prompt) // 4 + len(raw) // 4

                # Parse action; fall back to action_space.sample() on failure
                action: Any = action_space.sample() if action_space is not None else 0
                try:
                    m = re.search(r'"action"\s*:\s*([^\s,}]+)', raw)
                    if m:
                        act_val = m.group(1).strip().strip('"')
                        if action_space is not None and hasattr(action_space, "n"):
                            action = int(act_val) % int(action_space.n)
                        else:
                            try:
                                action = json.loads(act_val)
                            except Exception:
                                action = act_val
                except Exception:
                    pass  # keep random fallback

                step_out = env.step(action)
                obs = _ag_get(step_out, "observation", obs)
                total_reward += float(_ag_get(step_out, "reward", 0.0))
                terminated = bool(_ag_get(step_out, "terminated", False))
                truncated = bool(_ag_get(step_out, "truncated", False))
                steps += 1
                if terminated or truncated:
                    break

            elapsed = time.time() - t0
        finally:
            try:
                env.close()
            except Exception:
                pass

        return {
            "steps": steps,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "total_tokens": total_tokens,
            "elapsed_secs": elapsed,
        }

    # ── Run each requested mode and collect results ───────────────────────────
    sections: list[str] = [f"rl_llm_play_episode - {env_id}\n"]

    for mode in mode_list:
        use_lang = mode == "language"
        mode_label = "Language-translated observations" if use_lang else "Raw observations"

        if use_lang and translator is None:
            sections.append(
                f"Mode: {mode_label}\n"
                "  Skipped - no language translator registered for this environment.\n"
                "  Register one via rl_set_language_translator() first.\n"
            )
            continue

        try:
            r = await _run_mode(use_language=use_lang)
        except Exception as exc:
            sections.append(
                f"Mode: {mode_label}\n"
                f"  Error: {exc}\n"
            )
            continue

        if r["terminated"]:
            outcome = "terminated \u2714 (episode end reached - task solved)"
        elif r["truncated"]:
            outcome = "truncated (hit max_steps limit - did not naturally terminate)"
        else:
            outcome = f"incomplete (stopped after {r['steps']} steps)"

        sections.append(
            f"Mode: {mode_label}\n"
            f"  Steps:        {r['steps']}\n"
            f"  Total reward: {r['total_reward']:.4f}\n"
            f"  Outcome:      {outcome}\n"
            f"  Tokens used:  ~{r['total_tokens']:,}\n"
            f"  Wall time:    {r['elapsed_secs']:.1f}s\n"
        )

    return "\n".join(sections)
