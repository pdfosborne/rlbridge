"""
Automated RL experiment pipeline MCP tool for the RLIP plugin.

Tools: rl_experiment_process
"""

from __future__ import annotations

import threading
import time
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
from ._dashboard import dashboard as _dash, is_running as _dash_running, start_dashboard as _start_dashboard, dashboard_url as _dashboard_url
from ._tools_agents_utils import _build_agent, _render_policy_for_dashboard
from ._tools_agents_training import rl_train_agent


@mcp.tool()
def rl_experiment_process(
    env_id: str,
    agent_type: Optional[str] = None,
    instruction: str = "",
    n_episodes_baseline: Optional[int] = None,
    n_episodes_instruction: Optional[int] = None,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    top_k: Optional[int] = None,
    min_episode_visits: Optional[int] = None,
    sub_goal_threshold: Optional[float] = None,
    # Tabular Q hyper-parameters
    alpha: Optional[float] = None,
    gamma: Optional[float] = None,
    epsilon: Optional[float] = None,
    epsilon_min: Optional[float] = None,
    epsilon_decay: Optional[float] = None,
    # DQN / PPO shared
    hidden_size: Optional[int] = None,
    lr: Optional[float] = None,
) -> str:
    """
    Automated RL experiment pipeline — one call runs the full sequence in the
    background.  Returns a job_id immediately; poll with
    rl_get_training_result(job_id) until status is "done".

    **This is the DEFAULT tool for agent training.**  Use rl_train_agent()
    only when you need manual control (single phase, explicit match_id, or
    detailed hyper-parameter tuning).

    Pipeline stages (executed sequentially in a background thread):

    1. **Language translation check** — validates that a translator is
       registered for *env_id*.  Fails early if none is found.
    2. **Baseline training** — trains a fresh agent on raw observations
       while a LanguageTrackingWrapper records which language-described
       states are visited during successful episodes.
    3. **Language-state training** — trains a second agent whose
       observations are the translated language strings instead of raw
       state vectors.  Provides a direct comparison to the baseline.
    4. **Instruction derivation** — after baseline training, top-k states
       most correlated with success are scored and ranked.  Skipped when
       *instruction* is provided.
    5. **Instruction matching** — the top derived instruction (or the
       user-supplied *instruction*) is matched to the best language state
       in the observation cache; no new environment exploration is required.
    6. **Instruction-shaped training** — calls rl_train_agent(match_id=...)
       to train a fresh agent with sequential sub-goal reward shaping.
    7. **Instruction + language-state training** — repeats stage 6 with
       ``use_language_state=True`` so the agent sees translated language
       strings *and* benefits from sub-goal reward shaping simultaneously.

    Parameters
    ----------
    env_id:
        A registered RLIP environment ID, e.g. ``"Sailing-v0"``.
    agent_type:
        One of ``"tabular_q"``, ``"dqn"``, or ``"ppo"``.  Defaults to the
        environment's ``suggested_hyperparameters.agent_type`` when omitted,
        then falls back to ``"tabular_q"``.
    instruction:
        Optional natural-language goal.  When provided, stages 3–4 use this
        instruction directly instead of auto-deriving one from baseline.
    n_episodes_baseline:
        Episodes for the baseline phase.  Defaults to the environment's
        suggestion (``suggested_hyperparameters.n_episodes_baseline``) or 300.
    n_episodes_instruction:
        Episodes for the instruction-shaped phase.  Defaults to the
        environment's suggestion or 300.
    max_steps:
        Step cap per episode (both phases).  Defaults to the environment's
        suggestion or 200.
    seed:
        Optional reproducibility seed.
    top_k:
        Max derived instruction candidates (ignored when instruction= is
        provided).  Defaults to the environment's suggestion or 3.
    min_episode_visits:
        States visited in fewer than this many distinct episodes are excluded
        from derived candidates.  Defaults to suggestion or 2.
    sub_goal_threshold:
        Cosine similarity threshold to trigger the sub-goal bonus (0–1).
        Defaults to suggestion or 0.5.
    alpha / gamma / epsilon / epsilon_min / epsilon_decay:
        tabular_q hyper-parameters.  All default to environment suggestions
        when available, then RLIP global defaults.
    hidden_size / lr:
        dqn and ppo hyper-parameters.  Same defaulting strategy.

    Returns
    -------
    A job_id string.  Poll with ``rl_get_training_result(job_id)`` until the
    status is ``"done"``.  The final result includes both agent IDs, the
    instruction used, and clean evaluation metrics.
    """
    from ..environments.registry import registry as _env_registry  # noqa: PLC0415
    from ..language_translation import get_translator  # noqa: PLC0415

    try:
        _env_factory = _env_registry.get(env_id)
    except KeyError:
        return (
            f"Environment '{env_id}' is not registered.  "
            "Call rl_list_environments() to see available environments."
        )

    # ── Resolve hyperparameters (env suggestion → RLIP global default) ────────
    # Persisted override (from rl_update_suggested_hyperparameters) takes
    # precedence over the factory's built-in default.
    _hp = _hp_overrides.get(env_id) or _env_factory.env_info.suggested_hyperparameters
    _suggested_note = ""
    if _hp is not None:
        _suggested_note = (
            f"  Using suggested hyperparameters from '{env_id}'.\n"
            f"  Override any value by passing it explicitly.\n\n"
        )

    def _r(val, suggested, default):
        """Return val if explicitly provided, else suggested, else default."""
        return val if val is not None else (suggested if _hp is not None else default)

    agent_type   = _r(agent_type,   getattr(_hp, "agent_type",  None) if _hp else None, "tabular_q")
    n_episodes_baseline    = _r(n_episodes_baseline,    getattr(_hp, "n_episodes_baseline",   None), 300)
    n_episodes_instruction = _r(n_episodes_instruction, getattr(_hp, "n_episodes_instruction", None), 300)
    max_steps              = _r(max_steps,              getattr(_hp, "max_steps",              None), 200)
    top_k                  = _r(top_k,                  getattr(_hp, "top_k",                 None), 3)
    min_episode_visits     = _r(min_episode_visits,     getattr(_hp, "min_episode_visits",    None), 2)
    sub_goal_threshold     = _r(sub_goal_threshold,     getattr(_hp, "sub_goal_threshold",    None), 0.5)
    alpha                  = _r(alpha,                  getattr(_hp, "alpha",                 None), 0.1)
    gamma                  = _r(gamma,                  getattr(_hp, "gamma",                 None), 0.99)
    epsilon                = _r(epsilon,                getattr(_hp, "epsilon",               None), 1.0)
    epsilon_min            = _r(epsilon_min,            getattr(_hp, "epsilon_min",           None), 0.01)
    epsilon_decay          = _r(epsilon_decay,          getattr(_hp, "epsilon_decay",         None), 0.995)
    hidden_size            = _r(hidden_size,            getattr(_hp, "hidden_size",           None), 64)
    lr                     = _r(lr,                     getattr(_hp, "lr",                    None), 1e-3)

    agent_type = agent_type.lower().strip()
    if agent_type not in _AGENT_DESCRIPTIONS:
        return (
            f"Unknown agent type '{agent_type}'.  "
            f"Valid choices: {', '.join(_AGENT_DESCRIPTIONS)}.\n"
            "Call rl_list_agents() for details."
        )

    translator = _custom_translators.get(env_id) or get_translator(env_id)
    auto_translator_note = ""
    if translator is None:
        # Auto-install a str() fallback so the pipeline can proceed without
        # requiring a manually written translator.  The user can call
        # rl_set_translator_code() later for richer semantics.
        from ..language_translation.generator import GeneratedTranslator  # noqa: PLC0415
        from ..language_translation import TRANSLATORS  # noqa: PLC0415
        _fallback_code = (
            "def translate(state, *, legal_moves=None, action_history=None):\n"
            "    return str(state)\n"
        )
        translator = GeneratedTranslator(
            llm_fn=lambda p: "",
            env_id=env_id,
            rule_code=_fallback_code,
            refine_threshold=99_999,
        )
        _custom_translators[env_id] = translator
        TRANSLATORS[env_id] = translator
        auto_translator_note = (
            f"  Note: no translator registered for '{env_id}' — "
            f"a str() fallback was auto-installed.\n"
            f"  Call rl_set_translator_code() afterward for richer instruction matching.\n\n"
        )

    instruction = instruction.strip()
    top_k = max(1, min(top_k, 20))

    # Always ensure the dashboard is running so progress is visible
    dash_url = _start_dashboard()

    job_id = uuid.uuid4().hex[:12]
    _training_jobs[job_id] = {
        "status": "running",
        "agent_id": None,
        "env_id": env_id,
        "n_episodes": n_episodes_baseline + n_episodes_instruction,
        "progress_env": None,
        "phase": "baseline",
    }

    def _job() -> None:
        import sys as _sys
        import tqdm as _tqdm
        from ..instruction_following import (  # noqa: PLC0415
            LanguageTrackingWrapper,
            derive_instructions_from_training,
            build_sequential_instruction_following_protocol,
        )
        from ..environments.registry import registry as _reg  # noqa: PLC0415
        from ..language_translation import get_translator as _get_translator  # noqa: PLC0415

        _translator = _custom_translators.get(env_id) or _get_translator(env_id)
        factory = _reg.get(env_id)

        # ── Phase 1: Baseline training with language tracking ─────────────────
        env1 = factory.create()
        wrapper = LanguageTrackingWrapper(env1, translator=_translator)

        # Pre-allocate so the dashboard can be registered before training starts
        baseline_agent_id = uuid.uuid4().hex[:12]
        agent1 = _build_agent(
            agent_type, alpha, gamma,
            epsilon, epsilon_min, epsilon_decay,
            hidden_size, lr, seed,
        )

        if _dash_running():
            _dash.register(
                baseline_agent_id,
                agent_type=agent_type,
                env_id=env_id,
                n_episodes=n_episodes_baseline,
                use_language_state=False,
                uses_instructions=False,
                instructions=[],
            )

        class _TrackBar:
            """Minimal env shim: tqdm progress bar + LanguageTrackingWrapper."""

            def __init__(self, w: Any, n: int, dash_id: str, agent_ref: Any) -> None:
                self._w = w
                self._best = float("-inf")
                self._ep_r = 0.0
                self._calls = 0
                self._completed_episodes = 0
                self._dash_id = dash_id
                self._agent_ref = agent_ref
                self._bar = _tqdm.tqdm(
                    total=n,
                    desc=f"Phase 1 baseline — {agent_type} on {env_id}",
                    unit="ep", file=_sys.stderr, dynamic_ncols=True, leave=True,
                )

            def reset(self, seed: Any = None, options: Any = None) -> Any:
                if self._calls > 0:
                    if self._ep_r > self._best:
                        self._best = self._ep_r
                    self._bar.set_postfix(
                        last=f"{self._ep_r:.2f}", best=f"{self._best:.2f}"
                    )
                    if self._bar.n < self._bar.total:
                        self._bar.update(1)
                    if self._dash_id:
                        _dash.update(
                            self._dash_id,
                            completed=self._completed_episodes,
                            last_reward=self._ep_r,
                            epsilon=float(getattr(self._agent_ref, "epsilon", 0.0)),
                        )
                    self._ep_r = 0.0
                    self._completed_episodes += 1
                self._calls += 1
                return self._w.reset(seed=seed, options=options)

            def step(self, action: Any) -> Any:
                r = self._w.step(action)
                try:
                    self._ep_r += float(
                        r.reward if hasattr(r, "reward") else r.get("reward", 0.0)
                    )
                except Exception:
                    pass
                return r

            def close(self) -> None:
                self._bar.close()
                if self._dash_id:
                    _dash.finish(self._dash_id)
                self._w.close()

            @property
            def action_space(self) -> Any:
                return self._w.action_space

            @property
            def env_id(self) -> str:
                return getattr(self._w, "env_id", "")

            def __getattr__(self, name: str) -> Any:
                return getattr(self._w, name)

        track_env = _TrackBar(
            wrapper, n_episodes_baseline,
            dash_id=baseline_agent_id if _dash_running() else "",
            agent_ref=agent1,
        )
        _training_jobs[job_id]["progress_env"] = track_env

        try:
            result1 = agent1.train(
                track_env, n_episodes=n_episodes_baseline,
                max_steps=max_steps, seed=seed,
            )
        except Exception as exc:
            _training_jobs[job_id]["status"] = "failed"
            _training_jobs[job_id]["result"] = f"Baseline training failed: {exc}"
            try:
                track_env.close()
            except Exception:
                pass
            return
        try:
            track_env.close()
        except Exception:
            pass

        _trained_agents[baseline_agent_id] = {
            "agent":                agent1,
            "env_id":               env_id,
            "agent_type":           agent_type,
            "train_result":         result1,
            "use_language_state":   False,
            "best_episode_history": getattr(result1, "best_episode_history", []),
            "instructions_used":    [],
        }
        _training_jobs[job_id]["agent_id"] = baseline_agent_id

        if _dash_running():
            _render_policy_for_dashboard(
                baseline_agent_id, env_id,
                getattr(result1, "best_episode_history", []), max_steps,
            )

        # ── Phase 1b: Language-state training (for comparison) ────────────────
        _training_jobs[job_id]["phase"] = "language-state training"

        jobs_before_lang = set(_training_jobs.keys())
        rl_train_agent(
            agent_type=agent_type,
            env_id=env_id,
            n_episodes=n_episodes_baseline,
            max_steps=max_steps,
            seed=seed,
            use_language_state=True,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            hidden_size=hidden_size,
            lr=lr,
        )
        lang_job_id = next(iter(set(_training_jobs.keys()) - jobs_before_lang), None)

        if lang_job_id:
            while _training_jobs.get(lang_job_id, {}).get("status") == "running":
                time.sleep(1)
        lang_agent_id = (
            _training_jobs.get(lang_job_id, {}).get("agent_id")
            if lang_job_id else None
        )

        # ── Determine instruction ──────────────────────────────────────────────
        _training_jobs[job_id]["phase"] = "deriving instructions"

        if instruction:
            instructions_to_use = [instruction]
            instruction_source = f"user-provided: {instruction!r}"
        else:
            derived = derive_instructions_from_training(
                wrapper,
                top_k=top_k,
                min_episode_visits=min_episode_visits,
            )
            if not derived:
                _training_jobs[job_id]["status"] = "done"
                _training_jobs[job_id]["result"] = (
                    f"Baseline training complete ({n_episodes_baseline} episodes) but no "
                    f"instruction candidates could be derived (too few successful episodes "
                    f"or min_episode_visits={min_episode_visits} not satisfied).\n\n"
                    f"Baseline agent saved as agent_id='{baseline_agent_id}'.\n"
                    f"Mean reward: {result1.mean_reward:.4f}  "
                    f"Best: {result1.best_reward:.4f}\n\n"
                    "Try increasing n_episodes_baseline or lowering min_episode_visits, "
                    "or pass an explicit instruction= to skip derivation."
                )
                return
            instructions_to_use = [derived[0].instruction]
            instruction_source = f"auto-derived: {derived[0].instruction!r}"
            log.info("rl_experiment_process: derived instruction — %s", derived[0].instruction)

        # ── Match instruction to observation states ────────────────────────────
        _training_jobs[job_id]["phase"] = "matching instructions"

        try:
            env2 = factory.create()
            protocol = build_sequential_instruction_following_protocol(
                instructions_to_use,
                env2,
                translator=_translator,
                max_steps=max_steps,
                seed=seed,
            )
        except Exception as exc:
            _training_jobs[job_id]["status"] = "failed"
            _training_jobs[job_id]["result"] = (
                f"Instruction matching failed: {exc}\n\n"
                f"Baseline agent saved as agent_id='{baseline_agent_id}'."
            )
            return

        # Register protocol in the shared cache so rl_train_agent can resolve it
        synthetic_match_id = uuid.uuid4().hex[:12]
        _instruction_protocols[synthetic_match_id] = {
            "protocol":             protocol,
            "is_sequential":        True,
            "encoder_name":         "tfidf",
            "original_instruction": instructions_to_use[0],
            "instructions":         instructions_to_use,
            "match_summary":        {},
        }

        # ── Phase 2: Instruction-shaped training via rl_train_agent ───────────
        _training_jobs[job_id]["phase"] = "instruction training"

        jobs_before = set(_training_jobs.keys())
        rl_train_agent(
            agent_type=agent_type,
            env_id=env_id,
            n_episodes=n_episodes_instruction,
            max_steps=max_steps,
            seed=seed,
            match_id=synthetic_match_id,
            sub_goal_threshold=sub_goal_threshold,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            hidden_size=hidden_size,
            lr=lr,
        )
        new_keys = set(_training_jobs.keys()) - jobs_before
        phase2_job_id = next(iter(new_keys), None)

        if phase2_job_id is None:
            _training_jobs[job_id]["status"] = "failed"
            _training_jobs[job_id]["result"] = (
                f"Instruction-shaped training could not be started.\n\n"
                f"Baseline agent saved as agent_id='{baseline_agent_id}'."
            )
            return

        # Wait for phase 2 to finish
        while _training_jobs.get(phase2_job_id, {}).get("status") == "running":
            time.sleep(1)

        if _training_jobs.get(phase2_job_id, {}).get("status") == "failed":
            _training_jobs[job_id]["status"] = "failed"
            _training_jobs[job_id]["result"] = (
                f"Instruction-shaped training failed.\n\n"
                f"Baseline agent saved as agent_id='{baseline_agent_id}'.\n"
                + _training_jobs[phase2_job_id].get("result", "")
            )
            return

        instr_agent_id = _training_jobs[phase2_job_id].get("agent_id", "?")
        _training_jobs[job_id]["agent_id"] = instr_agent_id

        instr_entry = _trained_agents.get(instr_agent_id, {})
        _eval_mean = instr_entry.get("eval_mean")
        _eval_std = instr_entry.get("eval_std")
        eval_line = ""
        if _eval_mean is not None and _eval_std is not None:
            eval_line = f"  Clean eval (100 eps): mean={_eval_mean:.4f}  \u00b1{_eval_std:.4f}\n"

        train_result2 = instr_entry.get("train_result")
        phase2_reward_line = ""
        if train_result2 is not None:
            phase2_reward_line = (
                f"  Mean reward: {train_result2.mean_reward:.4f}  "
                f"Best: {train_result2.best_reward:.4f}\n"
            )

        # ── Phase 3: Instructions + language-state training ───────────────────
        _training_jobs[job_id]["phase"] = "instruction + language-state training"

        jobs_before_lang_instr = set(_training_jobs.keys())
        rl_train_agent(
            agent_type=agent_type,
            env_id=env_id,
            n_episodes=n_episodes_instruction,
            max_steps=max_steps,
            seed=seed,
            match_id=synthetic_match_id,
            sub_goal_threshold=sub_goal_threshold,
            use_language_state=True,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            hidden_size=hidden_size,
            lr=lr,
        )
        lang_instr_job_id = next(
            iter(set(_training_jobs.keys()) - jobs_before_lang_instr), None
        )

        if lang_instr_job_id:
            while _training_jobs.get(lang_instr_job_id, {}).get("status") == "running":
                time.sleep(1)

        lang_instr_agent_id = (
            _training_jobs.get(lang_instr_job_id, {}).get("agent_id")
            if lang_instr_job_id else None
        )

        lang_instr_entry = _trained_agents.get(lang_instr_agent_id, {}) if lang_instr_agent_id else {}
        _li_eval_mean = lang_instr_entry.get("eval_mean")
        _li_eval_std = lang_instr_entry.get("eval_std")
        lang_instr_eval_line = ""
        if _li_eval_mean is not None and _li_eval_std is not None:
            lang_instr_eval_line = (
                f"  Clean eval (100 eps): mean={_li_eval_mean:.4f}  \u00b1{_li_eval_std:.4f}\n"
            )
        train_result3 = lang_instr_entry.get("train_result")
        phase3_reward_line = ""
        if train_result3 is not None:
            phase3_reward_line = (
                f"  Mean reward: {train_result3.mean_reward:.4f}  "
                f"Best: {train_result3.best_reward:.4f}\n"
            )

        lang_entry = _trained_agents.get(lang_agent_id, {}) if lang_agent_id else {}
        lang_result = lang_entry.get("train_result")
        lang_reward_line = ""
        if lang_result is not None:
            lang_reward_line = (
                f"  Mean reward: {lang_result.mean_reward:.4f}  "
                f"Best: {lang_result.best_reward:.4f}\n"
            )
        lang_section = (
            f"\nPhase 1b \u2014 Language-state training ({n_episodes_baseline} episodes):\n"
            + (lang_reward_line or "  (no result)\n")
            + f"  Language agent_id: {lang_agent_id}\n"
        ) if lang_agent_id else ""

        phase3_section = (
            f"\nPhase 3 \u2014 Instruction + language-state training ({n_episodes_instruction} episodes):\n"
            + (phase3_reward_line or "  (no result)\n")
            + lang_instr_eval_line
            + f"  Instruction+lang agent_id: {lang_instr_agent_id}\n"
        ) if lang_instr_agent_id else ""

        compare_ids = [baseline_agent_id]
        if lang_agent_id:
            compare_ids.append(lang_agent_id)
        compare_ids.append(instr_agent_id)
        if lang_instr_agent_id:
            compare_ids.append(lang_instr_agent_id)

        best_agent_id = lang_instr_agent_id or instr_agent_id
        _final_dash_url = _dashboard_url() or dash_url
        result_text = (
            f"Experiment pipeline complete \u2014 {agent_type} on {env_id}\n\n"
            f"Dashboard: {_final_dash_url}\n\n"
            f"Phase 1 \u2014 Baseline training ({n_episodes_baseline} episodes):\n"
            f"  Mean reward: {result1.mean_reward:.4f}  Best: {result1.best_reward:.4f}\n"
            f"  Baseline agent_id: {baseline_agent_id}\n"
            + lang_section
            + f"\nInstruction used ({instruction_source}):\n"
            + "".join(f"  {i + 1}. {s}\n" for i, s in enumerate(instructions_to_use))
            + f"\nPhase 2 \u2014 Instruction-shaped training ({n_episodes_instruction} episodes):\n"
            + phase2_reward_line
            + eval_line
            + f"  Instruction agent_id: {instr_agent_id}\n"
            + phase3_section
            + f"\nUse rl_run_agent_episode(agent_id='{best_agent_id}') to evaluate the "
            f"best agent.\n"
            f"Use rl_create_training_report(agent_id='{best_agent_id}', "
            f"compare_agent_ids={compare_ids!r}) to compare all phases."
        )
        _training_jobs[job_id]["result"] = result_text
        _training_jobs[job_id]["status"] = "done"

    threading.Thread(target=_job, daemon=True).start()

    instr_note = (
        f"  Instruction:   {instruction!r}\n"
        if instruction
        else "  Instruction:   (auto-derived from baseline)\n"
    )
    return (
        f"Experiment pipeline started \u2014 {agent_type} on {env_id}\n\n"
        f"  Job ID:        {job_id}\n"
        f"{instr_note}"
        f"  Phase 1:       {n_episodes_baseline} baseline episodes\n"
        f"  Phase 2+3:     {n_episodes_instruction} instruction-shaped episodes each\n"
        f"  max_steps:     {max_steps}\n\n"
        + _suggested_note
        + auto_translator_note
        + f"Dashboard: {dash_url}  (open in browser for live reward curves)\n\n"
        f"Pipeline stages:\n"
        f"  1. Baseline training with language state tracking\n"
        f"  2. Language-state training (comparison agent)\n"
        f"  3. {'Matching provided instruction' if instruction else 'Deriving instructions from baseline'}\n"
        f"  4. Instruction-shaped training\n"
        f"  5. Instruction + language-state training\n\n"
        f"Call rl_get_training_result(job_id='{job_id}') to check progress and get the result."
    )
