"""
Default RL experiment pipeline for the RLIP MCP plugin.

Tool: rl_experiment_process

Runs the full compare-baseline → language → derive → match → instruct → evaluate
workflow in one background job. Poll with rl_get_training_result(job_id).
"""

from __future__ import annotations

import re
import sys
import threading
import time
import traceback
import uuid
from typing import Any, Optional

from ._prompts import AGENT_DESCRIPTIONS as _AGENT_DESCRIPTIONS
from ._state import (
    _custom_translators,
    _hp_overrides,
    _instruction_protocols,
    _trained_agents,
    _training_jobs,
    log,
    mcp,
)
from ._dashboard import (
    dashboard as _dash,
    dashboard_url as _dashboard_url,
    is_running as _dash_running,
    start_dashboard as _start_dashboard,
)
from ._tools_agents_utils import _build_agent, _render_policy_for_dashboard
from ._tools_agents_training import rl_train_agent
from ._tools_agents_eval import _run_clean_evaluation
from ._tools_saved_experiments import rl_save_experiment as _rl_save_experiment

STAGES_TOTAL = 8


# ── Progress wrapper (tqdm + dashboard) ───────────────────────────────────────


class _ProgressEnv:
    """Thin shim: tqdm episode bar + optional dashboard updates."""

    def __init__(
        self,
        env: Any,
        *,
        n_episodes: int,
        desc: str,
        dash_id: str = "",
        agent_ref: Any = None,
    ) -> None:
        import tqdm

        self._env = env
        self._dash_id = dash_id
        self._agent_ref = agent_ref
        self._best = float("-inf")
        self._ep_r = 0.0
        self._calls = 0
        self._completed = 0
        self._bar = tqdm.tqdm(
            total=n_episodes,
            desc=desc,
            unit="ep",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
        )

    def reset(self, seed: Any = None, options: Any = None) -> Any:
        if self._calls > 0:
            if self._ep_r > self._best:
                self._best = self._ep_r
            self._bar.set_postfix(last=f"{self._ep_r:.2f}", best=f"{self._best:.2f}")
            if self._bar.n < self._bar.total:
                self._bar.update(1)
            if self._dash_id:
                _dash.update(
                    self._dash_id,
                    completed=self._completed,
                    last_reward=self._ep_r,
                    epsilon=float(getattr(self._agent_ref, "epsilon", 0.0)),
                )
            self._ep_r = 0.0
            self._completed += 1
        self._calls += 1
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:
        out = self._env.step(action)
        try:
            r = out.reward if hasattr(out, "reward") else out.get("reward", 0.0)
            self._ep_r += float(r)
        except Exception:
            pass
        return out

    def close(self) -> None:
        self._bar.close()
        if self._dash_id:
            _dash.finish(self._dash_id)
        self._env.close()

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def env_id(self) -> str:
        return getattr(self._env, "env_id", "")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


# ── Pipeline helpers ────────────────────────────────────────────────────────────


def _ensure_translator(env_id: str) -> tuple[Any, str]:
    """Return (translator, note). Auto-install str() fallback when missing."""
    from ..language_translation import TRANSLATORS, get_translator  # noqa: PLC0415
    from ..language_translation.generator import GeneratedTranslator  # noqa: PLC0415

    translator = _custom_translators.get(env_id) or get_translator(env_id)
    if translator is not None:
        return translator, ""

    rule_code = (
        "def translate(state, *, legal_moves=None, action_history=None):\n"
        "    return str(state)\n"
    )
    translator = GeneratedTranslator(
        llm_fn=lambda _p: "",
        env_id=env_id,
        rule_code=rule_code,
        refine_threshold=99_999,
    )
    _custom_translators[env_id] = translator
    TRANSLATORS[env_id] = translator
    note = (
        f"  Note: no translator for '{env_id}' - installed str() fallback.\n"
        f"  Call rl_set_translator_code() for richer matching.\n\n"
    )
    return translator, note


def _merge_tracking_wrappers(primary: Any, secondary: Any | None, ep_offset: int) -> None:
    """Merge Phase-1b language visits into the baseline tracking wrapper."""
    if secondary is None:
        return
    try:
        for lang, ep_set in secondary._lang_episodes.items():
            primary._lang_episodes.setdefault(lang, set()).update(
                {e + ep_offset for e in ep_set}
            )
            if lang not in primary._lang_obs_sample and lang in secondary._lang_obs_sample:
                primary._lang_obs_sample[lang] = secondary._lang_obs_sample[lang]
        for lang, ep_set in secondary._lang_success_episodes.items():
            primary._lang_success_episodes.setdefault(lang, set()).update(
                {e + ep_offset for e in ep_set}
            )
    except Exception as exc:
        raise RuntimeError(f"Merging language tracking data failed: {exc}") from exc


def _seed_obs_cache(env_id: str, *wrappers: Any) -> int:
    """Populate _OBS_CACHE from LanguageTrackingWrapper samples."""
    from ..instruction_following import _OBS_CACHE  # noqa: PLC0415

    cache = _OBS_CACHE.setdefault(env_id, {})
    n_before = len(cache)
    for wrapper in wrappers:
        if wrapper is None:
            continue
        for lang, obs in getattr(wrapper, "_lang_obs_sample", {}).items():
            cache.setdefault(lang, obs)
    return len(cache) - n_before


def _format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _fail_experiment(
    job_id: str,
    stage: int,
    label: str,
    exc: BaseException | str,
    *,
    completed: dict[str, str] | None = None,
    extra: str = "",
) -> None:
    """Mark experiment failed with stage, error details, and traceback."""
    _training_jobs[job_id]["status"] = "failed"
    _training_jobs[job_id]["phase"] = f"[{stage}/{STAGES_TOTAL}] {label} - FAILED"
    _training_jobs[job_id]["stages_done"] = max(0, stage - 1)

    if isinstance(exc, BaseException):
        exc_type = type(exc).__name__
        exc_msg = str(exc) or "(no message)"
        tb = _format_exception(exc)
        log.exception("experiment failed at stage %d (%s)", stage, label)
    else:
        exc_type = "Error"
        exc_msg = str(exc)
        tb = ""
        log.error("experiment failed at stage %d (%s): %s", stage, label, exc_msg)

    lines = [
        f"Experiment FAILED at stage {stage}/{STAGES_TOTAL}: {label}",
        "",
        f"Error type: {exc_type}",
        f"Message:    {exc_msg}",
    ]
    if extra:
        lines.extend(["", extra.rstrip()])
    if completed:
        lines.extend(["", "Completed before failure:"])
        for key, val in completed.items():
            lines.append(f"  {key}: {val}")
    if tb:
        lines.extend(["", "Traceback:", tb.rstrip()])

    _training_jobs[job_id]["result"] = "\n".join(lines)


def _wait_subjob(
    parent_job_id: str,
    sub_job_id: str,
    *,
    timeout_secs: float,
) -> tuple[bool, str]:
    """
    Poll a rl_train_agent sub-job; forward progress_env to parent.

    Returns (ok, detail).  On failure *detail* explains timeout or sub-job error.
    """
    deadline = time.time() + timeout_secs
    while _training_jobs.get(sub_job_id, {}).get("status") == "running":
        penv = _training_jobs.get(sub_job_id, {}).get("progress_env")
        if penv is not None:
            _training_jobs[parent_job_id]["progress_env"] = penv
        if time.time() > deadline:
            return False, (
                f"Sub-job '{sub_job_id}' timed out after {timeout_secs:.0f}s "
                f"(still status=running)."
            )
        time.sleep(1)

    sub = _training_jobs.get(sub_job_id, {})
    if sub.get("status") == "failed":
        detail = sub.get("result") or sub.get("error") or "(no sub-job error text)"
        return False, f"Sub-job '{sub_job_id}' failed:\n{detail}"
    if sub.get("status") != "done":
        return False, (
            f"Sub-job '{sub_job_id}' ended with unexpected status="
            f"{sub.get('status')!r}."
        )
    return True, ""


def _parse_job_id(ret: str | None) -> str | None:
    m = re.search(r"Job ID:\s+(\w+)", ret or "")
    return m.group(1) if m else None


def _store_agent(
    agent_id: str,
    *,
    agent: Any,
    env_id: str,
    agent_type: str,
    train_result: Any,
    use_language_state: bool,
    instructions_used: list[str],
    match_id: str = "",
    eval_mean: float | None = None,
    eval_std: float | None = None,
) -> None:
    _trained_agents[agent_id] = {
        "agent": agent,
        "env_id": env_id,
        "agent_type": agent_type,
        "train_result": train_result,
        "use_language_state": use_language_state,
        "best_episode_history": getattr(train_result, "best_episode_history", []),
        "instructions_used": instructions_used,
        "match_id": match_id,
        "eval_mean": eval_mean,
        "eval_std": eval_std,
    }


def _eval_inline_agent(
    agent_id: str,
    *,
    env_factory: Any,
    env_id: str,
    agent: Any,
    use_language_state: bool,
    translator: Any,
    n_episodes: int,
    max_steps: int,
    seed: int | None,
) -> tuple[float | None, float | None]:
    n_eval = min(30, max(10, n_episodes // 10))
    mean, std, _ = _run_clean_evaluation(
        agent=agent,
        env_factory=env_factory,
        n_episodes=n_eval,
        max_steps=max_steps,
        use_language_state=use_language_state,
        translator=translator if use_language_state else None,
        env_id=env_id,
        seed=seed if seed is not None else 0,
    )
    entry = _trained_agents.get(agent_id, {})
    entry["eval_mean"] = mean
    entry["eval_std"] = std
    return mean, std


def _train_tracked(
    *,
    factory: Any,
    env_id: str,
    translator: Any,
    agent_type: str,
    agent_hp: dict[str, Any],
    n_episodes: int,
    max_steps: int,
    seed: int | None,
    use_language_state: bool,
    desc: str,
    parent_job_id: str,
    instructions_used: list[str],
) -> tuple[str, Any, Any, Any]:
    """
    Train one agent with LanguageTrackingWrapper on every run.

    Returns (agent_id, tracking_wrapper, train_result, agent).
    """
    from ..instruction_following import LanguageTrackingWrapper  # noqa: PLC0415
    from ._env_wrappers import _LangStateEnv  # noqa: PLC0415

    base = factory.create()
    tracker = LanguageTrackingWrapper(base, translator=translator)
    train_env: Any = tracker
    if use_language_state:
        train_env = _LangStateEnv(tracker, translator=translator, env_id=env_id)

    agent_id = uuid.uuid4().hex[:12]
    agent = _build_agent(
        agent_type,
        agent_hp["alpha"],
        agent_hp["gamma"],
        agent_hp["epsilon"],
        agent_hp["epsilon_min"],
        agent_hp["epsilon_decay"],
        agent_hp["hidden_size"],
        agent_hp["lr"],
        agent_hp["seed"],
    )

    if _dash_running():
        _dash.register(
            agent_id,
            agent_type=agent_type,
            env_id=env_id,
            n_episodes=n_episodes,
            use_language_state=use_language_state,
            uses_instructions=bool(instructions_used),
            instructions=instructions_used,
        )

    progress = _ProgressEnv(
        train_env,
        n_episodes=n_episodes,
        desc=desc,
        dash_id=agent_id if _dash_running() else "",
        agent_ref=agent,
    )
    _training_jobs[parent_job_id]["progress_env"] = progress

    result = agent.train(
        progress, n_episodes=n_episodes, max_steps=max_steps, seed=seed
    )
    try:
        progress.close()
    except Exception:
        pass

    _store_agent(
        agent_id,
        agent=agent,
        env_id=env_id,
        agent_type=agent_type,
        train_result=result,
        use_language_state=use_language_state,
        instructions_used=instructions_used,
    )
    if _dash_running():
        _render_policy_for_dashboard(
            agent_id, env_id, getattr(result, "best_episode_history", []), max_steps
        )
    return agent_id, tracker, result, agent


def _resolve_instructions(
    *,
    user_instruction: str,
    baseline_tracker: Any,
    lang_tracker: Any | None,
    n_episodes: int,
    top_k: int,
    min_episode_visits: int,
) -> tuple[list[str], str]:
    """Derive or accept instructions; always merge tracking data first."""
    from ..instruction_following import derive_instructions_from_training  # noqa: PLC0415

    _merge_tracking_wrappers(baseline_tracker, lang_tracker, n_episodes)

    if user_instruction:
        return [user_instruction], f"user-provided: {user_instruction!r}"

    derived = derive_instructions_from_training(
        baseline_tracker, top_k=top_k, min_episode_visits=min_episode_visits
    )
    if not derived and min_episode_visits > 1:
        derived = derive_instructions_from_training(
            baseline_tracker, top_k=top_k, min_episode_visits=1
        )
    if derived:
        instr = derived[0].instruction
        return [instr], f"auto-derived: {instr!r}"

    langs = list(getattr(baseline_tracker, "_lang_episodes", {}).keys())
    if langs:
        fallback = max(
            langs, key=lambda s: len(baseline_tracker._lang_episodes.get(s, set()))
        )
        return [fallback], f"fallback (most-visited): {fallback!r}"

    raise RuntimeError(
        "No language states were recorded during training. "
        "Register a translator for this environment or pass instruction=."
    )


def _format_reward_line(result: Any | None) -> str:
    if result is None:
        return "  (no training result)\n"
    return (
        f"  Mean reward: {result.mean_reward:.4f}  "
        f"Best: {result.best_reward:.4f}\n"
    )


def _format_eval_line(entry: dict[str, Any]) -> str:
    mean = entry.get("eval_mean")
    std = entry.get("eval_std")
    if mean is None or std is None:
        return ""
    return f"  Clean eval: mean={mean:.4f}  ±{std:.4f}\n"


# ── MCP tool ────────────────────────────────────────────────────────────────────


@mcp.tool()
def rl_experiment_process(
    env_id: str,
    agent_type: Optional[str] = None,
    instruction: str = "",
    n_episodes: Optional[int] = None,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    top_k: Optional[int] = None,
    min_episode_visits: Optional[int] = None,
    sub_goal_threshold: Optional[float] = None,
    alpha: Optional[float] = None,
    gamma: Optional[float] = None,
    epsilon: Optional[float] = None,
    epsilon_min: Optional[float] = None,
    epsilon_decay: Optional[float] = None,
    hidden_size: Optional[int] = None,
    lr: Optional[float] = None,
    n_envs: Optional[int] = None,
    user_objective: str = "",
    auto_save_experiment: bool = True,
) -> str:
    """
    **Default experiment process** - one call runs the full pipeline in the background.

    Poll ``rl_get_training_result(job_id)`` until ``status == "done"`` or
    ``status == "failed"``.  On failure the result includes the failing stage,
    error type/message, what completed earlier, and a full traceback.

    Pipeline (8 stages):

    1. **Dashboard** - start live reward tracking UI
    2. **Baseline** - train on raw observations; ``LanguageTrackingWrapper`` records
       translated state visits
    3. **Language** - train on translated observations (translator auto-created if needed)
    4. **Derive** - ``derive_instructions_from_training`` from combined visit logs
    5. **Match** - match instructions to observed states (obs cache from stages 2–3;
       no fresh exploration)
    6. **Instruction** - train with sub-goal shaping (``rl_train_agent`` + ``match_id``)
    7. **Instruction + language** - same shaping with language-state observations
    8. **Evaluate & render** - clean eval, compare agents, render policies, pick best

    Use ``rl_train_agent()`` only for manual single-phase training.
    """
    from ..environments.registry import registry as _env_registry  # noqa: PLC0415

    try:
        factory = _env_registry.get(env_id)
    except KeyError:
        return (
            f"Environment '{env_id}' is not registered.  "
            "Call rl_list_environments() to see available environments."
        )

    _hp = _hp_overrides.get(env_id) or factory.env_info.suggested_hyperparameters

    def _r(val: Any, suggested: Any, default: Any) -> Any:
        if val is not None:
            return val
        if _hp is not None and suggested is not None:
            return suggested
        return default

    agent_type = (_r(agent_type, getattr(_hp, "agent_type", None), "tabular_q") or "tabular_q")
    agent_type = str(agent_type).lower().strip()
    n_episodes = int(_r(n_episodes, getattr(_hp, "n_episodes", None), 300))
    max_steps = int(_r(max_steps, getattr(_hp, "max_steps", None), 200))
    top_k = max(1, min(int(_r(top_k, getattr(_hp, "top_k", None), 3)), 20))
    min_episode_visits = int(_r(min_episode_visits, getattr(_hp, "min_episode_visits", None), 2))
    sub_goal_threshold = float(_r(sub_goal_threshold, getattr(_hp, "sub_goal_threshold", None), 0.5))
    n_envs = int(_r(n_envs, getattr(_hp, "n_envs", None), 10))

    agent_hp = {
        "alpha": _r(alpha, getattr(_hp, "alpha", None), 0.1),
        "gamma": _r(gamma, getattr(_hp, "gamma", None), 0.99),
        "epsilon": _r(epsilon, getattr(_hp, "epsilon", None), 1.0),
        "epsilon_min": _r(epsilon_min, getattr(_hp, "epsilon_min", None), 0.01),
        "epsilon_decay": _r(epsilon_decay, getattr(_hp, "epsilon_decay", None), 0.995),
        "hidden_size": _r(hidden_size, getattr(_hp, "hidden_size", None), 64),
        "lr": _r(lr, getattr(_hp, "lr", None), 1e-3),
        "seed": seed,
    }

    if agent_type not in _AGENT_DESCRIPTIONS:
        return (
            f"Unknown agent type '{agent_type}'.  "
            f"Valid: {', '.join(_AGENT_DESCRIPTIONS)}."
        )

    user_instruction = instruction.strip()
    suggested_note = (
        f"  Using suggested hyperparameters from '{env_id}'.\n\n" if _hp else ""
    )

    translator, translator_note = _ensure_translator(env_id)
    dash_url = _start_dashboard()

    job_id = uuid.uuid4().hex[:12]
    _training_jobs[job_id] = {
        "status": "running",
        "agent_id": None,
        "env_id": env_id,
        "n_episodes": n_episodes,
        "progress_env": None,
        "phase": "[1/8] baseline training",
        "stages_total": STAGES_TOTAL,
        "stages_done": 0,
    }

    def _job() -> None:
        from ..instruction_following import (  # noqa: PLC0415
            build_sequential_instruction_following_protocol,
        )
        from ..environments.registry import registry as _reg  # noqa: PLC0415

        _translator = _custom_translators.get(env_id) or translator
        factory_local = _reg.get(env_id)
        sub_timeout = max(300.0, n_episodes * max_steps / 500 + 120)
        completed: dict[str, str] = {"env_id": env_id, "agent_type": agent_type}

        def _set_phase(n: int, label: str) -> None:
            _training_jobs[job_id]["phase"] = f"[{n}/{STAGES_TOTAL}] {label}"
            _training_jobs[job_id]["stages_done"] = n - 1

        try:
            # ── 2. Baseline (raw env + language tracking) ───────────────────
            _set_phase(2, "baseline training")
            baseline_id, baseline_tracker, result_baseline, agent_baseline = _train_tracked(
                factory=factory_local,
                env_id=env_id,
                translator=_translator,
                agent_type=agent_type,
                agent_hp=agent_hp,
                n_episodes=n_episodes,
                max_steps=max_steps,
                seed=seed,
                use_language_state=False,
                desc=f"Baseline - {agent_type} on {env_id}",
                parent_job_id=job_id,
                instructions_used=[],
            )
            completed["stage_2_baseline_agent_id"] = baseline_id
            completed["stage_2_baseline_mean_reward"] = (
                f"{result_baseline.mean_reward:.4f}"
            )

            # ── 3. Language-state training (tracking underneath) ────────────
            _set_phase(3, "language-state training")
            lang_id, lang_tracker, result_lang, _ = _train_tracked(
                factory=factory_local,
                env_id=env_id,
                translator=_translator,
                agent_type=agent_type,
                agent_hp=agent_hp,
                n_episodes=n_episodes,
                max_steps=max_steps,
                seed=seed,
                use_language_state=True,
                desc=f"Language - {agent_type} on {env_id}",
                parent_job_id=job_id,
                instructions_used=[],
            )
            completed["stage_3_language_agent_id"] = lang_id
            completed["stage_3_language_mean_reward"] = f"{result_lang.mean_reward:.4f}"

            # ── 4. Derive instructions ────────────────────────────────────────
            _set_phase(4, "deriving instructions")
            instructions_to_use, instruction_source = _resolve_instructions(
                user_instruction=user_instruction,
                baseline_tracker=baseline_tracker,
                lang_tracker=lang_tracker,
                n_episodes=n_episodes,
                top_k=top_k,
                min_episode_visits=min_episode_visits,
            )
            completed["stage_4_instruction_source"] = instruction_source
            completed["stage_4_instructions"] = " | ".join(instructions_to_use)
            log.info("experiment: instructions - %s", instruction_source)

            # ── 5. Match using observed states (no new exploration) ───────────
            _set_phase(5, "matching instructions to observed states")
            n_cached = _seed_obs_cache(env_id, baseline_tracker, lang_tracker)
            completed["stage_5_obs_cache_new_states"] = str(n_cached)
            log.info("experiment: obs cache seeded with %d new states", n_cached)

            match_env = factory_local.create()
            protocol = build_sequential_instruction_following_protocol(
                instructions_to_use,
                match_env,
                translator=_translator,
                max_steps=max_steps,
                seed=seed,
            )
            match_id = uuid.uuid4().hex[:12]
            _instruction_protocols[match_id] = {
                "protocol": protocol,
                "is_sequential": True,
                "encoder_name": "tfidf",
                "original_instruction": instructions_to_use[0],
                "instructions": instructions_to_use,
                "env_id": env_id,
                "match_summary": {},
            }
            completed["stage_5_match_id"] = match_id

            clean_n = min(30, max(10, n_episodes // 10))
            train_kw = dict(
                agent_type=agent_type,
                env_id=env_id,
                n_episodes=n_episodes,
                max_steps=max_steps,
                seed=seed,
                match_id=match_id,
                sub_goal_threshold=sub_goal_threshold,
                n_envs=n_envs,
                clean_eval_episodes=clean_n,
                clean_eval_timeout_secs=120,
                **{
                    k: agent_hp[k]
                    for k in (
                        "alpha", "gamma", "epsilon", "epsilon_min",
                        "epsilon_decay", "hidden_size", "lr",
                    )
                },
            )

            # ── 6. Instruction-shaped training ────────────────────────────────
            _set_phase(6, "instruction-shaped training")
            _training_jobs[job_id]["n_episodes"] = (
                n_episodes * n_envs if n_envs > 1 else n_episodes
            )

            p2_ret = rl_train_agent(**train_kw)
            p2_job = _parse_job_id(p2_ret)
            if p2_job is None:
                raise RuntimeError(
                    "rl_train_agent did not return a job id for instruction training.\n"
                    f"rl_train_agent returned:\n{p2_ret}"
                )

            ok2, detail2 = _wait_subjob(job_id, p2_job, timeout_secs=sub_timeout)
            if not ok2:
                raise RuntimeError(detail2)

            instr_id = _training_jobs[p2_job].get("agent_id")
            if not instr_id:
                raise RuntimeError(
                    f"Instruction training sub-job '{p2_job}' finished without agent_id."
                )
            instr_entry = _trained_agents.get(instr_id, {})
            instr_entry["instructions_used"] = list(instructions_to_use)
            instr_entry["match_id"] = match_id
            completed["stage_6_instruction_agent_id"] = instr_id

            # ── 7. Instruction + language training ────────────────────────────
            _set_phase(7, "instruction + language-state training")
            p3_ret = rl_train_agent(**train_kw, use_language_state=True)
            p3_job = _parse_job_id(p3_ret)
            if p3_job is None:
                raise RuntimeError(
                    "rl_train_agent did not return a job id for instruction+language training.\n"
                    f"rl_train_agent returned:\n{p3_ret}"
                )

            ok3, detail3 = _wait_subjob(job_id, p3_job, timeout_secs=sub_timeout)
            if not ok3:
                raise RuntimeError(detail3)

            lang_instr_id = _training_jobs[p3_job].get("agent_id")
            if not lang_instr_id:
                raise RuntimeError(
                    f"Instruction+language sub-job '{p3_job}' finished without agent_id."
                )
            li_entry = _trained_agents.get(lang_instr_id, {})
            li_entry["instructions_used"] = list(instructions_to_use)
            li_entry["match_id"] = match_id
            completed["stage_7_instruction_language_agent_id"] = lang_instr_id

            # ── 8. Evaluate, compare, render ──────────────────────────────────
            _set_phase(8, "evaluate, compare, and render")
            _eval_inline_agent(
                baseline_id,
                env_factory=factory_local,
                env_id=env_id,
                agent=agent_baseline,
                use_language_state=False,
                translator=_translator,
                n_episodes=n_episodes,
                max_steps=max_steps,
                seed=seed,
            )
            lang_agent = _trained_agents.get(lang_id, {}).get("agent")
            if lang_agent is None:
                raise RuntimeError(
                    f"Language agent '{lang_id}' missing from trained-agents cache."
                )
            _eval_inline_agent(
                lang_id,
                env_factory=factory_local,
                env_id=env_id,
                agent=lang_agent,
                use_language_state=True,
                translator=_translator,
                n_episodes=n_episodes,
                max_steps=max_steps,
                seed=seed,
            )

            all_ids = [baseline_id, lang_id, instr_id, lang_instr_id]
            for aid in all_ids:
                if _dash_running() and aid in _trained_agents:
                    hist = _trained_agents[aid].get("best_episode_history", [])
                    _render_policy_for_dashboard(aid, env_id, hist, max_steps)

            candidates: list[tuple[str, float | None, bool]] = []
            for aid in all_ids:
                entry = _trained_agents.get(aid, {})
                candidates.append(
                    (
                        aid,
                        entry.get("eval_mean"),
                        bool(entry.get("use_language_state", False)),
                    )
                )
            best_id, best_mean, best_lang = max(
                candidates,
                key=lambda x: x[1] if x[1] is not None else float("-inf"),
            )
            best_entry = _trained_agents.get(best_id, {})

            _training_jobs[job_id]["agent_id"] = best_id
            _training_jobs[job_id]["stages_done"] = STAGES_TOTAL
            _training_jobs[job_id]["phase"] = f"[{STAGES_TOTAL}/{STAGES_TOTAL}] complete"

            saved_section = ""
            if auto_save_experiment:
                objective = (
                    user_objective.strip()
                    or user_instruction
                    or " → ".join(instructions_to_use)
                )
                try:
                    save_out = _rl_save_experiment(
                        user_objective=objective,
                        env_id=env_id,
                        agent_id=best_id,
                        agent_type=agent_type,
                        instruction=" → ".join(instructions_to_use),
                        use_language_state=best_lang,
                        eval_mean=best_mean,
                        eval_std=best_entry.get("eval_std"),
                        notes=(
                            f"Default experiment pipeline. Source: {instruction_source}. "
                            f"Candidates: {', '.join(aid for aid, _, _ in candidates)}"
                        ),
                    )
                    m = re.search(r"Experiment ID:\s+(\S+)", save_out)
                    if m:
                        saved_section = (
                            f"\nSaved experiment: {m.group(1)}\n"
                            f"  Recall: rl_load_experiment(experiment_id='{m.group(1)}')\n"
                        )
                except Exception as exc:
                    log.warning("experiment: auto-save failed - %s", exc)

            compare_ids = list(all_ids)
            li_entry = _trained_agents.get(lang_instr_id, {})
            result_text = (
                f"Default experiment complete - {agent_type} on {env_id}\n\n"
                f"Dashboard: {_dashboard_url() or dash_url}\n"
                f"Instruction ({instruction_source}):\n"
                + "".join(f"  {i + 1}. {s}\n" for i, s in enumerate(instructions_to_use))
                + f"\nStage 2 - Baseline ({n_episodes} ep, language tracking):\n"
                + _format_reward_line(result_baseline)
                + _format_eval_line(_trained_agents.get(baseline_id, {}))
                + f"  agent_id: {baseline_id}\n"
                + f"\nStage 3 - Language observations ({n_episodes} ep):\n"
                + _format_reward_line(result_lang)
                + _format_eval_line(_trained_agents.get(lang_id, {}))
                + f"  agent_id: {lang_id}\n"
                + f"\nStage 6 - Instruction-shaped ({n_episodes} ep):\n"
                + _format_reward_line(instr_entry.get("train_result"))
                + _format_eval_line(instr_entry)
                + f"  agent_id: {instr_id}\n"
                + f"\nStage 7 - Instruction + language ({n_episodes} ep):\n"
                + _format_reward_line(li_entry.get("train_result"))
                + _format_eval_line(li_entry)
                + f"  agent_id: {lang_instr_id}\n"
                + f"\nBest agent (by clean eval): {best_id}"
                + (f"  mean={best_mean:.4f}" if best_mean is not None else "")
                + f"  use_language_state={best_lang}\n"
                + saved_section
                + f"\nCompare: rl_create_training_report(agent_id='{best_id}', "
                f"compare_agent_ids={compare_ids!r})\n"
                f"Run: rl_run_agent_episode(agent_id='{best_id}')"
            )
            _training_jobs[job_id]["status"] = "done"
            _training_jobs[job_id]["result"] = result_text

        except Exception as exc:
            stage_label = _training_jobs[job_id].get("phase", "unknown")
            stage_num = STAGES_TOTAL
            m = re.search(r"\[(\d+)/", str(stage_label))
            if m:
                stage_num = int(m.group(1))
            label = re.sub(r"^\[\d+/\d+\]\s*", "", str(stage_label))
            label = label.replace(" - FAILED", "").strip() or "unknown"
            _fail_experiment(
                job_id,
                stage_num,
                label,
                exc,
                completed=completed,
            )

    threading.Thread(target=_job, daemon=True).start()

    instr_note = (
        f"  Instruction: {user_instruction!r}\n"
        if user_instruction
        else "  Instruction: (derived from training)\n"
    )
    return (
        f"Default experiment started - {agent_type} on {env_id}\n\n"
        f"  Job ID: {job_id}\n"
        f"{instr_note}"
        f"  Episodes/phase: {n_episodes}  max_steps: {max_steps}\n"
        f"{suggested_note}"
        f"{translator_note}"
        f"Dashboard: {dash_url}\n\n"
        f"Stages (poll rl_get_training_result until status='done'):\n"
        f"  1. Dashboard\n"
        f"  2. Baseline + language tracking\n"
        f"  3. Language-state training + tracking\n"
        f"  4. Derive instructions from observed states\n"
        f"  5. Match instructions (obs cache, no re-exploration)\n"
        f"  6. Instruction-shaped training\n"
        f"  7. Instruction + language training\n"
        f"  8. Evaluate, compare, render policies\n\n"
        f"Call rl_get_training_result(job_id='{job_id}') every 30–60s until done."
    )
