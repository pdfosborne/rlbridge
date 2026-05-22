"""
RL agent training MCP tools for the rlbridge plugin.

Tools: rl_train_agent, rl_get_training_result
"""

from __future__ import annotations

import json
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ._dashboard import dashboard as _dash, is_running as _dash_running
from ._env_wrappers import _LangStateEnv, _SequentialShapedEnv, _ShapedEnv
from ._prompts import AGENT_DESCRIPTIONS as _AGENT_DESCRIPTIONS
from ._state import (
    _env_agents_registry_path,
    _env_cache_dir,
    _env_plan_db_path,
    _custom_translators,
    _instruction_protocols,
    _trained_agents,
    _training_jobs,
    log,
    mcp,
)
from ._tools_agents_io import (
    _package_trained_agent,
)
from ._tools_agents_utils import (
    _build_agent,
    _collect_training_instructions,
    _render_policy_for_dashboard,
)
from ._tools_agents_eval import _run_clean_evaluation

class _ProgressEnv:
    """
    Thin env wrapper that renders a tqdm progress bar on sys.stderr during
    agent.train().  Intercepts reset() to advance the bar once per episode
    and step() to track the current episode reward for the postfix display.

    Written to stderr so it never touches the MCP stdout channel.
    """

    def __init__(
        self,
        env: Any,
        n_episodes: int,
        agent_type: str,
        env_id: str,
        *,
        dashboard_agent_id: str = "",
        agent_ref: Any = None,
    ) -> None:
        import tqdm
        self._env = env
        self._n_episodes = n_episodes
        self._best_reward = float("-inf")
        self._ep_reward = 0.0
        self._reset_calls = 0
        self._completed_episodes = 0
        self._dashboard_agent_id = dashboard_agent_id
        self._agent_ref = agent_ref
        self._bar = tqdm.tqdm(
            total=n_episodes,
            desc=f"Training {agent_type} on {env_id}",
            unit="ep",
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
        )

    def reset(self, seed: Any = None, options: Any = None) -> Any:
        if self._reset_calls > 0:
            # A previous episode just ended - commit its reward and advance.
            if self._ep_reward > self._best_reward:
                self._best_reward = self._ep_reward
            self._bar.set_postfix(
                last=f"{self._ep_reward:.2f}",
                best=f"{self._best_reward:.2f}",
            )
            n = min(1, self._n_episodes - self._bar.n)
            if n > 0:
                self._bar.update(n)
            _finished_reward = self._ep_reward
            self._ep_reward = 0.0
            self._completed_episodes += 1
            if self._dashboard_agent_id:
                _dash.update(
                    self._dashboard_agent_id,
                    completed=self._completed_episodes,
                    last_reward=_finished_reward,
                    epsilon=float(getattr(self._agent_ref, "epsilon", 0.0)),
                )
        self._reset_calls += 1
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        try:
            r = result.reward if hasattr(result, "reward") else result.get("reward", 0.0)
            self._ep_reward += float(r)
        except Exception:
            pass
        return result

    def close(self) -> None:
        self._bar.close()
        if self._dashboard_agent_id and self._reset_calls > 0:
            # Commit the last episode whose reward was accumulated but never
            # flushed (there is no subsequent reset() call after the final ep).
            self._completed_episodes += 1
            _dash.update(
                self._dashboard_agent_id,
                completed=min(self._completed_episodes, self._n_episodes),
                last_reward=self._ep_reward,
                epsilon=float(getattr(self._agent_ref, "epsilon", 0.0)),
            )
            _dash.finish(self._dashboard_agent_id)
        elif self._dashboard_agent_id:
            _dash.finish(self._dashboard_agent_id)
        self._env.close()

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def env_id(self) -> str:
        return getattr(self._env, "env_id", "")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class _ParallelProgressProxy:
    """
    Lightweight proxy that acts as the *progress_env* slot in ``_training_jobs``
    when parallel training is active.  Each sub-worker calls ``_increment()``
    after every completed episode so that ``rl_get_training_result`` can report
    an accurate aggregate progress percentage.
    """

    def __init__(self, n_total_episodes: int) -> None:
        self._completed_episodes: int = 0
        self._lock = threading.Lock()
        self._n_total = n_total_episodes

    def _increment(self) -> None:
        with self._lock:
            self._completed_episodes += 1

    def close(self) -> None:
        pass


@mcp.tool()

def rl_train_agent(
    agent_type: str,
    env_id: str,
    n_episodes: int = 300,
    max_steps: int = 200,
    seed: Optional[int] = None,
    match_id: str = "",
    sub_goal_bonus: float = 0.0,
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
    # Parallel environments
    n_envs: int = 10,
    # Post-training clean evaluation
    clean_eval_episodes: int = 100,
    clean_eval_timeout_secs: int = 180,
) -> str:
    """
    Train an RL agent on a registered environment (manual / advanced mode).

    **For most use-cases, prefer** ``rl_experiment_process()`` **instead.**
    ``rl_experiment_process`` handles language setup, baseline training,
    instruction derivation and shaped training automatically in one call.

    Use this tool only when you need manual control: running a single
    phase, supplying an explicit ``match_id`` from a previous
    ``rl_match_instruction`` call, or tuning hyper-parameters without
    the auto-pipeline.

    Call rl_list_agents() first to see which agent types are available and
    when to use each one.

    Parameters
    ----------
    agent_type:
        One of "tabular_q", "dqn", or "ppo".
    env_id:
        A registered rlbridge environment ID, e.g. "Sailing-v0".
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
        when match_id is provided).  Set to 0.0 (default) to auto-scale:
        ``max_reward / (100 × n_sub_goals)`` where *max_reward* is inferred
        from the environment's reward range and *n_sub_goals* is the number
        of matched states.  Pass an explicit positive value to override.
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
        Discount factor - shared by all agents.
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
    n_envs:
        Number of independent parallel environments to train on simultaneously.
        Each worker gets its own environment instance and a unique seed offset.
        After all workers finish, the agent with the highest best-episode reward
        is selected.  Combined episode statistics from all workers are reported.
        Set to 1 to disable parallel training (original single-env behaviour).
    clean_eval_episodes:
        Number of post-training clean-evaluation episodes (fixed weights, no
        instruction rewards).  Set to 0 to skip clean evaluation.
    clean_eval_timeout_secs:
        Hard timeout for clean evaluation. If exceeded, evaluation is skipped so
        the training job can still complete.

    Returns
    -------
    A training summary and an agent_id for use with rl_run_agent_episode().
    """
    from ..rl_agents import TabularQAgent, DQNAgent, PPOAgent  # noqa: PLC0415
    from ..environments.registry import registry as _env_registry  # noqa: PLC0415

    agent_type = agent_type.lower().strip()
    resolved_bonus: Optional[float] = None  # set when sub-goal shaping is active
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
    # Cap matched states per stage: environments with numeric observations
    # produce empty TF-IDF vocabularies, so all observed states tie at
    # similarity 0.0 and ALL get included in matched_states.  Without this cap
    # every training step iterates over potentially thousands of zero vectors,
    # making training look permanently stuck.
    _MAX_STAGE_LANGS = 20
    shaping_summary = ""
    if match_id:
        entry = _instruction_protocols.get(match_id)
        if entry is None:
            return (
                f"match_id '{match_id}' not found.  "
                "Call rl_match_instruction() first to generate a valid match_id."
            )
        protocol = entry["protocol"]
        is_sequential = bool(entry.get("is_sequential"))
        n_sub_goals = 0
        # None tells _ShapedEnv to auto-scale; explicit >0 overrides.
        effective_bonus: Optional[float] = None if sub_goal_bonus == 0.0 else sub_goal_bonus
        # Resolve encoder factory from the match entry (defaults to tfidf).
        _encoder_name = entry.get("encoder_name", "tfidf")
        from ..instruction_matching import get_encoder as _get_encoder  # noqa: PLC0415
        def _encoder_factory(_enc_name=_encoder_name):
            return _get_encoder(_enc_name)
        if is_sequential:
            stage_languages: list[list[str]] = []
            for m in getattr(protocol, "matches", []):
                langs = [lg for lg, _obs, _sc in getattr(m, "matched_states", [])]
                if not langs and getattr(m, "matched_language", None):
                    langs = [m.matched_language]
                if langs:
                    stage_languages.append(langs[:_MAX_STAGE_LANGS])
            if not stage_languages:
                return (
                    f"match_id '{match_id}' does not have valid sequential stages. "
                    "Re-run rl_match_instruction() or rl_match_sequential_instructions()."
                )
            n_sub_goals = sum(len(s) for s in stage_languages)
            shaped_env = _SequentialShapedEnv(
                env,
                stage_languages=stage_languages,
                bonus=effective_bonus,
                threshold=sub_goal_threshold,
                translator=getattr(protocol, "translate", True),
                env_id=env_id,
                encoder_factory=_encoder_factory,
            )
            shaped_env._ensure_encoders()
            resolved_bonus = shaped_env._bonus
            env = shaped_env
            shaping_summary = (
                f"\n  Sub-goal shaping: ON (sequential)  (match_id={match_id})\n"
                f"  Stages:           {len(stage_languages)}\n"
                f"  Sub-goals total:  {n_sub_goals} state(s)\n"
                f"  Bonus (auto-scaled): {resolved_bonus:.6g} / threshold={sub_goal_threshold}"
            )
        else:
            n_sub_goals = len(protocol._all_sub_goal_languages)
            # Wrap the environment so that step() injects the similarity bonus.
            shaped_env = _ShapedEnv(
                env,
                sub_goal_language=protocol.sub_goal_language,
                sub_goal_languages=[
                    lg for lg in protocol._all_sub_goal_languages
                    if lg != protocol.sub_goal_language
                ],
                bonus=effective_bonus,
                threshold=sub_goal_threshold,
                translator=protocol.translate,
                env_id=env_id,
                encoder_factory=_encoder_factory,
            )
            # Trigger encoder/bonus resolution now so we can report the value.
            shaped_env._ensure_encoder()
            resolved_bonus = shaped_env._bonus
            env = shaped_env
            shaping_summary = (
                f"\n  Sub-goal shaping: ON  (match_id={match_id})\n"
                f"  Sub-goals:        {n_sub_goals} state(s)\n"
                f"  Primary:          {protocol.sub_goal_language!r}\n"
                f"  Bonus (auto-scaled): {resolved_bonus:.6g} / threshold={sub_goal_threshold}"
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

    # ── Capture parameters for parallel worker env / agent creation ──────────
    # Computed once here (after shaping and lang-state wrappers are validated)
    # so that each parallel worker can build its own identically-configured env.
    _par_shaping_mode: str = ""          # "sequential" | "single" | ""
    _par_stage_languages: list[list[str]] = []
    _par_sub_goal_language: str = ""
    _par_extra_sg_languages: list[str] = []
    _par_translate: Any = True
    _par_encoder_name: str = "tfidf"
    _par_translator: Any = None

    if match_id and match_id in _instruction_protocols:
        _par_entry = _instruction_protocols[match_id]
        _par_protocol = _par_entry["protocol"]
        _par_is_sequential = bool(_par_entry.get("is_sequential"))
        _par_encode_name_val = _par_entry.get("encoder_name", "tfidf")
        _par_encoder_name = _par_encode_name_val
        _par_translate = getattr(_par_protocol, "translate", True)
        if _par_is_sequential:
            _par_shaping_mode = "sequential"
            for _par_m in getattr(_par_protocol, "matches", []):
                _par_langs = [lg for lg, _obs, _sc in getattr(_par_m, "matched_states", [])]
                if not _par_langs and getattr(_par_m, "matched_language", None):
                    _par_langs = [_par_m.matched_language]
                if _par_langs:
                    # Same cap as stage_languages above - avoids O(N) per step.
                    _par_stage_languages.append(_par_langs[:_MAX_STAGE_LANGS])
        else:
            _par_shaping_mode = "single"
            _par_sub_goal_language = _par_protocol.sub_goal_language
            _par_extra_sg_languages = [
                lg for lg in _par_protocol._all_sub_goal_languages
                if lg != _par_protocol.sub_goal_language
            ]

    if use_language_state:
        from ..language_translation import get_translator as _gtr_par  # noqa: PLC0415
        _par_translator = _custom_translators.get(env_id) or _gtr_par(env_id)

    def _build_worker_env() -> Any:
        """Create a fresh env with all configured wrappers for a parallel worker."""
        from ..environments.registry import registry as _ereg  # noqa: PLC0415
        from ..instruction_matching import get_encoder as _genc  # noqa: PLC0415
        _we = _ereg.get(env_id).create()
        if _par_shaping_mode == "sequential" and _par_stage_languages:
            _enc_name = _par_encoder_name
            def _enc_factory_seq(_n=_enc_name): return _genc(_n)
            _wse = _SequentialShapedEnv(
                _we,
                stage_languages=_par_stage_languages,
                bonus=resolved_bonus,   # already scaled; never None at call time
                threshold=sub_goal_threshold,
                translator=_par_translate,
                env_id=env_id,
                encoder_factory=_enc_factory_seq,
            )
            _wse._ensure_encoders()
            _we = _wse
        elif _par_shaping_mode == "single" and _par_sub_goal_language:
            _enc_name = _par_encoder_name
            def _enc_factory_sgl(_n=_enc_name): return _genc(_n)
            _wse = _ShapedEnv(
                _we,
                sub_goal_language=_par_sub_goal_language,
                sub_goal_languages=_par_extra_sg_languages,
                bonus=resolved_bonus,   # already scaled; never None at call time
                threshold=sub_goal_threshold,
                translator=_par_translate,
                env_id=env_id,
                encoder_factory=_enc_factory_sgl,
            )
            _wse._ensure_encoder()
            _we = _wse
        if use_language_state and _par_translator is not None:
            _we = _LangStateEnv(_we, translator=_par_translator, env_id=env_id)
        return _we

    def _build_worker_agent(worker_idx: int) -> Any:
        """Create a fresh agent instance for a parallel training worker."""
        from ..rl_agents import TabularQAgent as _TQ, DQNAgent as _DQ, PPOAgent as _PP  # noqa: PLC0415
        _ws = None if seed is None else seed + worker_idx
        if agent_type == "tabular_q":
            return _TQ(
                n_actions=2, alpha=alpha, gamma=gamma, epsilon=epsilon,
                epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, seed=_ws,
            )
        elif agent_type == "dqn":
            return _DQ(
                hidden_size=hidden_size, lr=lr, gamma=gamma, epsilon=epsilon,
                epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                buffer_size=buffer_size, batch_size=batch_size,
                target_update_freq=target_update_freq, seed=_ws,
            )
        else:
            return _PP(
                hidden_size=hidden_size, lr_actor=lr, lr_critic=lr_critic,
                gamma=gamma, lam=lam, clip_eps=clip_eps, n_steps=n_steps,
                ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size, seed=_ws,
            )

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

    agent_id = uuid.uuid4().hex[:12]

    # Register with the live dashboard if it is running
    if _dash_running():
        dash_instructions: list[str] = []
        if match_id and match_id in _instruction_protocols:
            _p = _instruction_protocols[match_id]["protocol"]
            seq_instrs = getattr(_p, "instructions", None)
            if isinstance(seq_instrs, list) and seq_instrs:
                dash_instructions.extend(str(s) for s in seq_instrs[:5])
            else:
                instr = getattr(_p, "instruction", "")
                if instr:
                    dash_instructions.append(str(instr))
        _dash.register(
            agent_id,
            agent_type=agent_type,
            env_id=env_id,
            n_episodes=n_episodes,
            use_language_state=use_language_state,
            uses_instructions=bool(match_id),
            instructions=dash_instructions,
        )

    progress_env = _ProgressEnv(
        env,
        n_episodes=n_episodes,
        agent_type=agent_type,
        env_id=env_id,
        dashboard_agent_id=agent_id if _dash_running() else "",
        agent_ref=agent,
    )

    job_id = uuid.uuid4().hex[:12]
    _par_proxy: Any = None
    if n_envs > 1:
        _par_proxy = _ParallelProgressProxy(n_envs * n_episodes)
        # Replace the progress_env slot with the proxy so rl_get_training_result
        # can report aggregate progress across all workers.
        _progress_env_for_job: Any = _par_proxy
    else:
        _progress_env_for_job = progress_env

    _training_jobs[job_id] = {
        "status": "running",
        "agent_id": agent_id,
        "env_id": env_id,
        "n_episodes": n_episodes * n_envs if n_envs > 1 else n_episodes,
        "progress_env": _progress_env_for_job,
    }

    def _job() -> None:
        try:
            _job_inner()
        except Exception as _top_exc:
            log.exception("rl_train_agent: unhandled exception in background job %s", job_id)
            _training_jobs[job_id]["status"] = "failed"
            _training_jobs[job_id]["result"] = (
                f"Training job failed (unhandled error): {_top_exc}"
            )

    def _job_inner() -> None:
        nonlocal agent
        _parallel_summary = ""

        if n_envs > 1:
            # ── Parallel training across n_envs independent workers ───────────
            import tqdm as _tqdm  # noqa: PLC0415
            _par_lock = threading.Lock()
            _worker_results: list[Any] = [None] * n_envs
            _par_bar = _tqdm.tqdm(
                total=n_envs * n_episodes,
                desc=f"Training {n_envs}x {agent_type} on {env_id}",
                unit="ep",
                file=sys.stderr,
                dynamic_ncols=True,
                leave=True,
            )

            def _sub_worker(widx: int) -> None:
                try:
                    _ws = None if seed is None else seed + widx
                    _we = _build_worker_env()
                    _wa = _build_worker_agent(widx)
                except Exception as exc:
                    log.warning("Parallel training worker %d setup failed: %s", widx, exc)
                    return

                _ep_r: list[float] = [0.0]
                _resets: list[int] = [0]

                class _CntEnv:
                    def reset(self_, seed: Any = None, options: Any = None) -> Any:
                        if _resets[0] > 0:
                            with _par_lock:
                                _par_proxy._increment()
                                _par_bar.update(1)
                            _ep_r[0] = 0.0
                        _resets[0] += 1
                        return _we.reset(seed=seed, options=options)

                    def step(self_, action: Any) -> Any:
                        out = _we.step(action)
                        try:
                            r = out.reward if hasattr(out, "reward") else out.get("reward", 0.0)
                            _ep_r[0] += float(r)
                        except Exception:
                            pass
                        return out

                    def close(self_) -> None:
                        if _resets[0] > 0:
                            with _par_lock:
                                _par_proxy._increment()
                                _par_bar.update(1)
                        _we.close()

                    @property
                    def action_space(self_) -> Any:
                        return _we.action_space

                    @property
                    def env_id(self_) -> str:
                        return getattr(_we, "env_id", env_id)

                    def __getattr__(self_, name: str) -> Any:
                        return getattr(_we, name)

                try:
                    wr = _wa.train(
                        _CntEnv(),
                        n_episodes=n_episodes,
                        max_steps=max_steps,
                        seed=_ws,
                    )
                    _worker_results[widx] = (_wa, wr)
                except Exception as exc:  # pragma: no cover
                    log.warning("Parallel training worker %d failed: %s", widx, exc)

            # Generous per-worker timeout: at least 5 min, scales with workload.
            _worker_timeout = max(300, n_episodes * max_steps // 1000 + 60)
            _sub_threads = [
                threading.Thread(target=_sub_worker, args=(i,), daemon=True)
                for i in range(n_envs)
            ]
            for _t in _sub_threads:
                _t.start()
            for _t in _sub_threads:
                _t.join(timeout=_worker_timeout)
            _par_bar.close()

            # Warn about workers that didn't complete within the timeout.
            _timed_out = [i for i, _t in enumerate(_sub_threads) if _t.is_alive()]
            if _timed_out:
                log.warning(
                    "rl_train_agent: %d/%d parallel workers timed out after %ss: %s",
                    len(_timed_out), n_envs, _worker_timeout, _timed_out,
                )

            _valid = [(wa, wr) for wa, wr in _worker_results if wr is not None]
            if not _valid:
                _training_jobs[job_id]["status"] = "failed"
                _training_jobs[job_id]["result"] = "All parallel training workers failed."
                return

            # Pick best agent by highest single-episode reward seen during training
            _best_wa, result = max(_valid, key=lambda x: x[1].best_reward)
            agent = _best_wa

            # Aggregate episode rewards across all workers for combined statistics
            _all_ep_rewards = [r for _, wr in _valid for r in wr.episode_rewards]
            _comb_mean = sum(_all_ep_rewards) / len(_all_ep_rewards) if _all_ep_rewards else 0.0
            _parallel_summary = (
                f"\n\n  Parallel workers:  {len(_valid)}/{n_envs} succeeded"
                f"\n  Combined episodes: {len(_all_ep_rewards)}"
                f"  (combined mean={_comb_mean:.4f})"
                f"\n  Best worker:       best_reward={result.best_reward:.4f}"
            )

        else:
            # ── Single-env training (original path) ───────────────────────────
            try:
                result = agent.train(
                    progress_env,
                    n_episodes=n_episodes,
                    max_steps=max_steps,
                    seed=seed,
                )
            except Exception as exc:
                _training_jobs[job_id]["status"] = "failed"
                _training_jobs[job_id]["result"] = f"Training failed: {exc}"
                try:
                    progress_env.close()
                except Exception:
                    pass
                return

            try:
                progress_env.close()
            except Exception:
                pass

        instructions_used = _collect_training_instructions(match_id)

        _trained_agents[agent_id] = {
            "agent":                agent,
            "env_id":               env_id,
            "agent_type":           agent_type,
            "best_episode_history": result.best_episode_history,
            "use_language_state":   use_language_state,
            "train_result":         result,
            "instructions_used":    instructions_used,
            "training_config": {
                "n_episodes": n_episodes,
                "max_steps": max_steps,
                "seed": seed,
                "match_id": match_id or None,
                "sub_goal_bonus": resolved_bonus,
                "sub_goal_threshold": sub_goal_threshold if match_id else None,
                "use_language_state": use_language_state,
                "alpha": alpha if agent_type == "tabular_q" else None,
                "gamma": gamma,
                "epsilon": epsilon if agent_type in {"tabular_q", "dqn"} else None,
                "epsilon_min": epsilon_min if agent_type in {"tabular_q", "dqn"} else None,
                "epsilon_decay": epsilon_decay if agent_type in {"tabular_q", "dqn"} else None,
                "hidden_size": hidden_size if agent_type in {"dqn", "ppo"} else None,
                "lr": lr if agent_type in {"dqn", "ppo"} else None,
                "buffer_size": buffer_size if agent_type == "dqn" else None,
                "batch_size": batch_size if agent_type == "dqn" else None,
                "target_update_freq": target_update_freq if agent_type == "dqn" else None,
                "lr_critic": lr_critic if agent_type == "ppo" else None,
                "lam": lam if agent_type == "ppo" else None,
                "clip_eps": clip_eps if agent_type == "ppo" else None,
                "n_steps": n_steps if agent_type == "ppo" else None,
                "ppo_epochs": ppo_epochs if agent_type == "ppo" else None,
                "mini_batch_size": mini_batch_size if agent_type == "ppo" else None,
            },
            "match_id":             match_id or None,
            "sub_goal_language":    (
                getattr(_instruction_protocols[match_id]["protocol"], "sub_goal_language", None)
                if match_id and match_id in _instruction_protocols else None
            ),
            "sub_goal_bonus":       resolved_bonus,
            "sub_goal_threshold":   sub_goal_threshold if match_id else None,
            "instruction":          (
                (
                    " -> ".join(getattr(_instruction_protocols[match_id]["protocol"], "instructions", [])[:5])
                    if getattr(_instruction_protocols[match_id]["protocol"], "instructions", None)
                    else getattr(_instruction_protocols[match_id]["protocol"], "instruction", None)
                )
                if match_id and match_id in _instruction_protocols else None
            ),
            "n_subgoals":           (
                (
                    sum(len(getattr(m, "matched_states", []) or [getattr(m, "matched_language", "")])
                        for m in getattr(_instruction_protocols[match_id]["protocol"], "matches", []))
                    if getattr(_instruction_protocols[match_id]["protocol"], "matches", None)
                    else len(getattr(_instruction_protocols[match_id]["protocol"], "_all_sub_goal_languages", []))
                )
                if match_id and match_id in _instruction_protocols else None
            ),
            "best_match_observation": (
                (_instruction_protocols.get(match_id, {}).get("match_summary") or {}).get("best_match_observation")
                if match_id else None
            ),
            "best_match_similarity": (
                (_instruction_protocols.get(match_id, {}).get("match_summary") or {}).get("best_match_similarity")
                if match_id else None
            ),
            "original_instruction": (
                _instruction_protocols.get(match_id, {}).get("original_instruction")
                if match_id else None
            ),
        }

        if _dash_running():
            _render_policy_for_dashboard(agent_id, env_id, result.best_episode_history, max_steps)

        # ── Clean evaluation (fixed weights, no instruction rewards) ───────────
        _eval_mean: float | None = None
        _eval_std: float | None = None
        _eval_rewards: list[float] = []
        if clean_eval_episodes > 0:
            _training_jobs[job_id]["phase"] = "post-training clean evaluation"
        try:
            from ..environments.registry import registry as _eval_env_registry  # noqa: PLC0415
            from ..language_translation import get_translator as _get_translator  # noqa: PLC0415

            _eval_factory = _eval_env_registry.get(env_id)
            _eval_agent = _trained_agents[agent_id]["agent"]
            _eval_translator = (
                _custom_translators.get(env_id) or _get_translator(env_id)
                if use_language_state else None
            )
            if clean_eval_episodes > 0:
                _eval_payload: dict[str, Any] = {}
                _eval_error: dict[str, Exception] = {}

                def _eval_runner() -> None:
                    try:
                        _mean, _std, _rewards = _run_clean_evaluation(
                            agent=_eval_agent,
                            env_factory=_eval_factory,
                            n_episodes=clean_eval_episodes,
                            max_steps=max_steps,
                            use_language_state=use_language_state,
                            translator=_eval_translator,
                            env_id=env_id,
                            seed=seed if seed is not None else 0,
                        )
                        _eval_payload["mean"] = _mean
                        _eval_payload["std"] = _std
                        _eval_payload["rewards"] = _rewards
                    except Exception as _exc:
                        _eval_error["exc"] = _exc

                _eval_thread = threading.Thread(target=_eval_runner, daemon=True)
                _eval_thread.start()
                _eval_thread.join(timeout=max(1, int(clean_eval_timeout_secs)))

                if _eval_thread.is_alive():
                    log.warning(
                        "rl_train_agent: clean evaluation timed out after %ss for agent %s",
                        clean_eval_timeout_secs,
                        agent_id,
                    )
                elif "exc" in _eval_error:
                    raise _eval_error["exc"]
                else:
                    _eval_mean = _eval_payload.get("mean")
                    _eval_std = _eval_payload.get("std")
                    _eval_rewards = _eval_payload.get("rewards", [])
                    _trained_agents[agent_id]["eval_mean"] = _eval_mean
                    _trained_agents[agent_id]["eval_std"] = _eval_std
                    _trained_agents[agent_id]["eval_rewards"] = _eval_rewards
        except Exception as _eval_exc:
            log.warning("rl_train_agent: clean evaluation skipped due to error: %s", _eval_exc)
            pass

        artifact_path: str | None = None
        artifact_warning = ""
        archive_path, archive_error = _package_trained_agent(agent_id, _trained_agents[agent_id])
        if archive_path is not None:
            artifact_path = str(archive_path)
            _trained_agents[agent_id]["artifact_archive"] = artifact_path
        elif archive_error:
            artifact_warning = f"\n  Artifact package: FAILED ({archive_error})"

        # Update plan database if this agent was trained with an instruction
        if match_id and match_id in _instruction_protocols and _eval_mean is not None:
            try:
                from ..instruction_plan_db import get_plan_database  # noqa: PLC0415
                from ._state import _env_plan_db_path  # noqa: PLC0415

                _proto_entry = _instruction_protocols[match_id]
                _original_instruction = (
                    _proto_entry.get("original_instruction")
                    or _proto_entry.get("match_summary", {}).get("instruction")
                    or ""
                )
                if _original_instruction:
                    _plan_db = get_plan_database(env_id, plan_path=str(_env_plan_db_path(env_id)))
                    _plan_db.update_eval_reward(
                        instruction=_original_instruction,
                        match_id=match_id,
                        eval_reward=_eval_mean,
                        training_reward=result.best_reward,
                        agent_id=agent_id,
                        agent_type=agent_type,
                        n_episodes=n_episodes,
                    )
            except Exception:
                pass

        try:
            _match_summary = (_instruction_protocols.get(match_id, {}).get("match_summary") or {}) if match_id else {}
            _reg_entry = {
                "agent_id":             agent_id,
                "agent_type":           agent_type,
                "env_id":               env_id,
                "created_at":           datetime.now().isoformat(),
                "instruction":          _trained_agents[agent_id].get("original_instruction")
                                        or _trained_agents[agent_id].get("instruction"),
                "sub_steps":            (_instruction_protocols.get(match_id, {}).get("instructions") or [])
                                        if match_id else [],
                "eval_reward":          _eval_mean,
                "eval_std":             _eval_std,
                "eval_n_episodes":      len(_eval_rewards) if _eval_rewards else None,
                "training_best_reward": result.best_reward,
                "match_similarity":     _trained_agents[agent_id].get("best_match_similarity"),
                "matched_language":     _match_summary.get("best_match_language"),
                "n_episodes":           n_episodes,
                "artifact_path":        artifact_path,
            }
            _reg_path = _env_agents_registry_path(env_id)
            _reg_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                _reg_data = json.loads(_reg_path.read_text(encoding="utf-8")) if _reg_path.exists() else {}
            except Exception:
                _reg_data = {}
            _reg_data[agent_id] = _reg_entry
            _reg_path.write_text(json.dumps(_reg_data, indent=2), encoding="utf-8")
        except Exception:
            pass

        eval_summary = ""
        if _eval_mean is not None and _eval_std is not None:
            eval_summary = (
                f"\n  Clean evaluation ({clean_eval_episodes} eps, fixed weights, no instruction rewards):\n"
                f"    mean reward = {_eval_mean:.4f}  ±  {_eval_std:.4f}  (std)\n"
            )

        summary = (
            f"Training complete - {agent_type} on {env_id}{shaping_summary}{lang_state_summary}"
            f"{_parallel_summary}\n\n"
            f"  {result}\n\n"
            f"  Mean reward (last 10 %): {result.last_n_mean:.4f}\n"
            f"{eval_summary}"
            f"  Agent ID: {agent_id}\n\n"
        )
        if artifact_path:
            summary += f"  Artifact package: {artifact_path}\n\n"
        if artifact_warning:
            summary += f"{artifact_warning}\n"
        summary += f"Use rl_run_agent_episode(agent_id='{agent_id}') to evaluate the agent."

        _training_jobs[job_id]["result"] = summary
        _training_jobs[job_id]["status"] = "done"

    threading.Thread(target=_job, daemon=True).start()

    _envs_line = f"  Envs:      {n_envs} parallel\n" if n_envs > 1 else ""
    return (
        f"Training started - {agent_type} on {env_id}{shaping_summary}{lang_state_summary}\n\n"
        f"  Agent ID:  {agent_id}\n"
        f"  Job ID:    {job_id}\n"
        f"  Episodes:  {n_episodes}  max_steps={max_steps}\n"
        f"{_envs_line}\n"
        f"Training runs in the background with no time limits.\n"
        f"Call rl_get_training_result(job_id='{job_id}') to check progress and get the result."
    )



@mcp.tool()
def rl_get_training_result(job_id: str) -> str:
    """
    Check the status of a background training job started by rl_train_agent
    or rl_experiment_process.

    Training runs in a background thread with no time limits so that long
    training runs are never interrupted by tool timeouts.

    **MANDATORY BEHAVIOUR**: keep calling this tool every 30–60 seconds until
    the status is "done".  When status is still "running", call it again —
    do NOT stop, summarise partial progress, or present results early.
    Only when the full result is returned (status == "done") should you
    present results to the user.

    Parameters
    ----------
    job_id:
        The job_id returned by rl_train_agent() or rl_experiment_process().

    Returns
    -------
    If still running: episode progress and a reminder to keep polling.
    If complete: the full training result summary.
    If failed: the error message.
    """
    job = _training_jobs.get(job_id)
    if not job:
        active = list(_training_jobs.keys())
        return (
            f"No training job found with ID '{job_id}'.\n"
            f"Active job IDs: {active if active else 'none'}"
        )
    if job["status"] == "running":
        progress_env = job.get("progress_env")
        done = getattr(progress_env, "_completed_episodes", 0) if progress_env else 0
        total = job.get("n_episodes", "?")
        pct = f"{done / total * 100:.0f}%" if isinstance(total, int) and total > 0 else "?"
        phase = job.get("phase", "")
        phase_line = f"  Phase:    {phase}\n" if phase else ""
        stages_done = job.get("stages_done", 0)
        stages_total = job.get("stages_total")
        stages_line = (
            f"  Stages:   {stages_done}/{stages_total} complete\n"
            if stages_total is not None else ""
        )
        agent_line = "  Agent ID: (not available until status is 'done')\n"
        return (
            f"Training in progress: {done}/{total} episodes ({pct})\n"
            f"{phase_line}"
            f"{stages_line}"
            f"{agent_line}"
            f"  Env:      {job.get('env_id', '?')}\n\n"
            f"** NOT DONE - call rl_get_training_result(job_id='{job_id}') again in ~30s. **\n"
            f"Do NOT report results or consider the experiment finished until status is 'done'."
        )
    return job.get("result", "No result available.")
