"""
LLM prompt templates and agent descriptions for RLIP MCP plugin tools.

Each function builds the full prompt string for a specific tool/call-site.
Dynamic values (env_id, instruction text, pre-formatted observation blocks,
etc.) are passed as arguments so the prompts themselves remain readable and
easy to edit without hunting through tool implementation files.

Constants
---------
AGENT_DESCRIPTIONS
    Human-readable descriptions of each trainable agent type, used by
    rl_list_agents() and validated in rl_train_agent / rl_experiment_process.

EXPERIMENT_SAVE_GUIDE
    Decision checklist shown to the LLM to determine when to call
    rl_save_experiment() and what fields to populate.

Call-site labels
----------------
SYSTEM: FastMCP(instructions=...)
    rlip_system_prompt()

TOOL: rl_match_instruction  (also used by rl_match_sequential_instructions)
    decompose_instruction_vocab_prompt()

TOOL: rl_train_and_derive_instructions
    decompose_instruction_simple_prompt()
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# SYSTEM: FastMCP(instructions=...) — delivered to the LLM for every session
# ---------------------------------------------------------------------------

def rlip_system_prompt() -> str:
    """
    Full system-level context prompt injected into every RLIP MCP session.

    This is the single authoritative description of what RLIP is, how to
    think with it, and the canonical tool workflows.  It is returned as a
    plain string so it can be read, tested, and edited independently of the
    FastMCP construction in ``_state.py``.
    """
    return (
        # ── What RLIP is ──────────────────────────────────────────────────
        "RLIP is a platform for automating sequential decision-making tasks "
        "through reinforcement learning.  You — the LLM — are the architect: "
        "you design environments, define how states translate into language, "
        "specify goals as natural-language instructions, and direct RL agents "
        "that learn to pursue those goals through trial and error.  The result "
        "is a self-contained, reproducible RL pipeline that runs entirely "
        "inside this tool layer.\n\n"

        # ── How to think about environment design ─────────────────────────
        "ENVIRONMENT DESIGN PRINCIPLES\n"
        "A good RLIP environment is self-contained: its observation space, "
        "action space, transition dynamics, and reward signal are fully "
        "determined by the environment's own code — no external process or "
        "human interaction is required at runtime.  When wrapping a task:\n"
        "  1. OBSERVATIONS must capture everything an agent needs to decide — "
        "state variables that distinguish meaningfully different situations.\n"
        "  2. ACTIONS must be the atomic decisions the agent can take — "
        "discrete choices or continuous controls, kept as small as the task "
        "allows.\n"
        "  3. REWARD must be dense enough for learning but aligned with the "
        "true objective — sparse terminal rewards are hard to learn from; "
        "add shaped intermediate rewards only where semantically justified.\n"
        "  4. EPISODE BOUNDS must be finite — every episode must eventually "
        "terminate or truncate so the agent can accumulate experience.\n"
        "  5. LANGUAGE TRANSLATION is the bridge between raw observations "
        "and goals expressed in natural language.  Write a translate(obs) "
        "function that converts any observation into a concise, descriptive "
        "English sentence.  Good translations mention the task-relevant "
        "entities, their positions or states, and the current objective "
        "status (e.g. 'boat heading north, wind from east, target 3 cells '). "
        "Translations should be distinct — two observations that call for "
        "different actions should produce different strings.\n\n"

        # ── Automation philosophy ─────────────────────────────────────────
        "AUTOMATING SEQUENTIAL TASKS WITH RLIP\n"
        "Any process that can be described as a loop — observe a state, "
        "choose an action, observe the outcome, repeat — can be automated "
        "with RLIP.  Useful patterns:\n"
        "  • GOAL-DIRECTED NAVIGATION: encode position/orientation as obs, "
        "movement primitives as actions, distance-to-goal as reward.  "
        "A language translator lets you specify the goal as plain text.\n"
        "  • PROCEDURE AUTOMATION: model a multi-step procedure as a "
        "sequential environment where completing each step unlocks the next. "
        "Use rl_experiment_process(instruction=...) to shape rewards toward "
        "the correct ordering automatically.\n"
        "  • OPTIMISATION UNDER CONSTRAINTS: wrap an optimisation problem "
        "as an RL env where the agent adjusts parameters each step and "
        "reward reflects objective improvement.  PPO handles continuous "
        "action spaces well here.\n"
        "  • GAME / SIMULATION CONTROL: wrap any Gymnasium-compatible "
        "simulation — rl_build_environment registers it in one call.\n\n"

        # ── Tool workflows (unchanged from previous system prompt) ─────────
        "Storage layout:\n"
        "  • ~/.rlip/  (cache) - custom env definitions, catalog, language-"
        "translation source, and instruction data.  Managed automatically.\n"
        "  • <cwd>/rlip_results/  (local saves) - policy render GIFs, "
        "training-report PNGs, and trained-agent ZIP packages.\n\n"

        "Custom environment workflow:\n"
        "1. rl_build_environment(env_id, gym_env_id, description, tags)\n"
        "2. rl_sample_states_for_translation(env_id)\n"
        "3. rl_set_translator_code(env_id, python_code)\n"
        "4. rl_translate_state(env_id, state) - verify quality.\n"
        "5. rl_load_cached_environments() - restore at startup.\n\n"

        "Instruction-following workflow:\n"
        "1. rl_match_instruction(env_id, instruction) - returns match_id.\n"
        "2. rl_instruction_run_episode(match_id) - sub-goal-shaped episode.\n\n"

        "RL agent training workflow:\n"
        "1. rl_list_agents() - tabular_q / dqn / ppo guidance.\n"
        "2. rl_experiment_process(agent_type, env_id) - DEFAULT.  Full pipeline: "
        "baseline + language tracking → instruction derivation → shaped "
        "training.  Pass instruction= to specify a goal.  Returns job_id.\n"
        "   • rl_train_agent() only for manual control (custom match_id, "
        "explicit hyper-parameters, no auto-pipeline).\n"
        "3. rl_get_training_result(job_id) - MUST be called repeatedly every\n"
        "   30–60 s until status is 'done'.  Do NOT stop polling after the\n"
        "   first call or report results while status is still 'running'.\n"
        "4. rl_run_agent_episode(agent_id) - one evaluation episode.\n"
        "5. rl_render_policy_overlay(env_id, agent_id=...) - composite PNG.\n"
        "6. rl_evaluate_agent(agent_id) - mean/std reward, percentiles, "
        "outcome fractions over 100 clean episodes.\n"
        "7. rl_create_training_report(agent_id, compare_agent_ids=[...]) - "
        "comparative PNG: reward convergence, eval bar chart, metadata.\n\n"

        "IMPORTANT — training is asynchronous: rl_experiment_process(), "
        "rl_train_agent(), and rl_train_and_derive_instructions() all return "
        "immediately with a job_id.  ALWAYS call rl_get_training_result(job_id) "
        "repeatedly until status is 'done' — the pipeline has multiple stages "
        "and a single poll is never sufficient.  Do NOT present results, "
        "summarise progress, or stop polling until status == 'done'.\n\n"

        "IMPORTANT — comparing agents: use identical n_episodes, max_steps, "
        "seed, gamma, and other shared hyper-parameters across every training "
        "call so differences reflect the agent/configuration, not budget.\n\n"

        "MANDATORY PLANNING RULE: call rl_get_instruction_plan(env_id) as the "
        "FIRST action on any environment.  Do not call rl_match_instruction() "
        "or rl_train_and_derive_instructions() before reading the plan.  Use "
        "stored eval rewards and derived scores to pick the next instruction "
        "rather than repeating something already tried.\n\n"

        "INSTRUCTION QUALITY RULE: only commit to an instruction that is "
        "likely to produce a trainable sub-goal:\n"
        "  • Prefer instructions whose BestEval reward in the plan DB is "
        "clearly above the random-policy baseline.\n"
        "  • Prefer instructions with a high similarity score (≥ 0.4) — a "
        "low score means the instruction text does not match any observed "
        "environment state well and the shaped reward signal will be weak.\n"
        "  • If rl_get_instruction_plan() shows an instruction that was tried "
        "before and produced poor eval rewards (close to or below random), do "
        "NOT repeat it — choose or derive a different instruction instead.\n"
        "  • After training, if rl_evaluate_agent() shows the agent is still "
        "weak, check whether the instruction itself is the bottleneck: call "
        "rl_get_instruction_plan() again and switch to the next-best "
        "instruction (highest BestEval not yet tried, or run "
        "rl_train_and_derive_instructions() to generate new candidates).  "
        "Only conclude the task is hard after at least two distinct "
        "instructions have been attempted and both failed evaluation.\n\n"

        "Also call rl_list_trained_agents(env_id) early in any session — if a "
        "saved agent matches the goal AND has a strong eval reward (clearly "
        "above the random-policy baseline), it is worth reusing or continuing "
        "training from.  Do NOT reuse a weak or unvalidated agent; start fresh "
        "with rl_experiment_process() instead.  When a promising saved agent "
        "exists, prefer rl_load_agent(agent_id=...) followed by "
        "rl_train_agent(...) to continue training from the loaded weights, "
        "rather than discarding accumulated learning.\n\n"

        "CONVERGENCE AND VALIDATION REQUIREMENT: after any training run, "
        "verify the agent actually works before reporting success:\n"
        "  1. rl_evaluate_agent(agent_id) — check mean reward is meaningfully "
        "above random and that the success/completion fraction is satisfactory.\n"
        "  2. rl_run_agent_episode(agent_id) — confirm at least one test episode "
        "reaches the goal / completes the task successfully.\n"
        "  3. If the agent fails both checks, first try continuing training "
        "(rl_train_agent with the same agent_id).  If a second training run "
        "still fails, apply the INSTRUCTION QUALITY RULE above: switch to a "
        "better instruction before starting fresh with a new agent.  Do not "
        "report a trained agent as successful until it passes evaluation.\n\n"

        "Combined instruction + training (preferred):\n"
        "rl_experiment_process(agent_type, env_id, instruction=instruction)\n\n"

        "Auto-derived instruction (no instruction needed):\n"
        "rl_experiment_process(agent_type, env_id)  # derives instruction automatically\n\n"

        "Instruction planning database: every match and training run is "
        "recorded in ~/.rlip/environments/<env>/instruction_plan.json.  "
        "rl_get_instruction_plan(env_id) shows all tried instructions, "
        "sub-steps, eval rewards, similarity scores, and usage history.  "
        "Always read it before any planning decision.\n\n"

        "SAVING AND RECALLING EXPERIMENTS\n"
        "At the start of every session, call rl_list_experiments() to check "
        "whether the user's objective has already been solved.  If a matching "
        "experiment with a strong eval reward is found, use "
        "rl_load_experiment(experiment_id=...) to recover the full "
        "configuration, then rl_load_agent(artifact_path=...) to restore the "
        "agent without re-training.\n"
        "After validating a new agent (rl_evaluate_agent passes, "
        "rl_run_agent_episode reaches the goal), always call "
        "rl_save_experiment() to persist the confirmed configuration.  "
        "Include the original user objective verbatim, the instruction used, "
        "the eval metrics, and notes explaining what was tried and why this "
        "configuration was selected.  Pass prompt_log= to capture the key "
        "conversation turns for reproducibility."
    )


# ---------------------------------------------------------------------------
# Agent type descriptions (shown to user / LLM by rl_list_agents)
# ---------------------------------------------------------------------------

AGENT_DESCRIPTIONS: dict[str, str] = {
    "tabular_q": (
        "Tabular Q-learning \u2013 lookup-table Q-learning with \u03b5-greedy exploration. "
        "Best for small discrete observation spaces (e.g. Sailing-v0 text strings). "
        "Fast to train, exact, but does not generalise to unseen states."
    ),
    "dqn": (
        "Deep Q-Network (DQN) \u2013 two-hidden-layer neural net with experience replay "
        "and a target network.  Works on any flat-vector or text observation; "
        "observations are encoded to a numeric vector automatically.  "
        "Good balance of speed and expressiveness."
    ),
    "ppo": (
        "Proximal Policy Optimisation (PPO) \u2013 actor-critic policy-gradient method "
        "with GAE advantage estimation and clipped surrogate objective.  "
        "Robust and sample-efficient; suitable for longer training runs."
    ),
}


# ---------------------------------------------------------------------------
# Experiment-save guidance (used in the system prompt and tool docstring)
# ---------------------------------------------------------------------------

EXPERIMENT_SAVE_GUIDE: str = (
    "SAVING A CONFIRMED EXPERIMENT\n"
    "After verifying the agent passes evaluation, call rl_save_experiment() to\n"
    "persist the confirmed configuration.  Use this checklist:\n\n"
    "  REQUIRED\n"
    "  • user_objective - the user's original goal, verbatim.\n"
    "  • env_id         - the environment the agent was trained on.\n"
    "  • agent_id       - the best agent's ID from the training result.\n"
    "  • agent_type     - 'tabular_q', 'dqn', or 'ppo'.\n\n"
    "  RECOMMENDED (add whenever available)\n"
    "  • instruction        - the instruction that shaped training.\n"
    "  • use_language_state - True if the agent saw translated obs strings.\n"
    "  • eval_mean / eval_std - from rl_evaluate_agent().\n"
    "  • notes              - why this configuration was chosen; what was tried\n"
    "    first; any observations about instruction quality or training stability.\n"
    "  • prompt_log         - key conversation turns that led to this config\n"
    "    (a list of {\"role\": ..., \"content\": ...} dicts).\n\n"
    "  OPTIONAL\n"
    "  • name       - short human-readable label (auto-generated if omitted).\n"
    "  • save_local - True to also write a copy to <cwd>/rlip_results/…\n\n"
    "After saving, confirm to the user with the experiment_id so they can\n"
    "recall it in future sessions with rl_load_experiment().\n"
)


# ---------------------------------------------------------------------------
# TOOL: rl_match_instruction
# TOOL: rl_match_sequential_instructions
# ---------------------------------------------------------------------------

def decompose_instruction_vocab_prompt(
    env_id: str,
    instruction: str,
    lang_block: str,
    vocab_block: str,
) -> str:
    """
    Prompt used by ``_decompose_instruction_with_llm`` in
    ``_tools_instruction.py`` to break a high-level instruction into concrete,
    ordered sub-steps.

    Uses an explicit vocabulary lexicon built from sampled environment state
    descriptions so the LLM is forced to use exact observable phrases rather
    than abstract domain jargon.

    Parameters
    ----------
    env_id:
        The environment identifier (e.g. ``"CartPole-v1"``).
    instruction:
        The raw user/LLM instruction to decompose.
    lang_block:
        Pre-formatted block of sampled translated language strings produced by
        the environment's language translator (one per line, indented with
        ``  - ``).  ~20 evenly-spaced entries from the full translated set.
    vocab_block:
        Pre-formatted block of recurring vocabulary clauses extracted from
        the translated language strings in *obs_block* (one per line, indented
        with ``  • ``).  Capped at ~60 entries by the caller.
    """
    return (
        f"You are setting up reward shaping for a reinforcement learning agent "
        f"in the environment \"{env_id}\".\n\n"
        f"The user's instruction is:\n  \"{instruction}\"\n\n"
        f"These are a representative sample of translated language strings the environment's "
        f"language translator can produce — use these as vocabulary examples (the full set may be larger):\n"
        f"{lang_block}\n\n"
        f"Recurring vocabulary clauses extracted from the descriptions above "
        f"(state, position, orientation, and condition phrases you MUST reuse verbatim):\n"
        f"{vocab_block}\n\n"
        f"Break the instruction into 2-5 concrete, ordered sub-steps the agent "
        f"must achieve in sequence. STRICT REQUIREMENTS:\n"
        f"1. Every sub-step MUST use EXACT phrases copied from the vocabulary "
        f"clauses above — do not paraphrase or invent new terms.\n"
        f"2. Do NOT use abstract domain jargon. If the instruction uses shorthand "
        f"(e.g. a named maneuver or game action), rewrite it as one or more "
        f"observable states drawn directly from the vocabulary above.\n"
        f"3. The first sub-step must describe the START of the episode.\n"
        f"4. Each sub-step must describe an observable state or transition that "
        f"can be matched directly to one of the state descriptions listed above.\n"
        f"5. No duplicates. At most 5 sub-steps.\n\n"
        f"Output ONLY a numbered list, one sub-step per line, no extra text."
    )


# ---------------------------------------------------------------------------
# TOOL: rl_train_and_derive_instructions
# ---------------------------------------------------------------------------

def decompose_instruction_simple_prompt(
    env_id: str,
    instruction: str,
    lang_block: str,
) -> str:
    """
    Prompt used by ``_decompose_instruction_with_llm`` in
    ``_tools_derived_instructions.py`` to break a high-level instruction into
    ordered, observable sub-steps during the derived-instruction training loop.

    Parameters
    ----------
    env_id:
        The environment identifier.
    instruction:
        The high-level instruction to decompose.
    lang_block:
        Pre-formatted block of sampled translated language strings produced by
        the environment's language translator (one per line, indented with
        ``  - ``).  ~20 evenly-spaced entries from the full translated set.
    """
    return (
        f"You are helping set up sequential reward shaping for RL in environment '{env_id}'.\n\n"
        f"High-level instruction:\n  '{instruction}'\n\n"
        f"Observed environment language states:\n{lang_block}\n\n"
        "Your task is to decide whether this instruction describes a SINGLE goal or a "
        "SEQUENCE of genuinely distinct goals, then output accordingly.\n\n"
        "RULES — read carefully before responding:\n"
        "1. A sub-step must represent a COMPLETE, INDEPENDENTLY MEANINGFUL environment state "
        "that the agent must physically reach — e.g. 'the boat is heading north', "
        "'the agent is adjacent to the target', 'the pole is nearly vertical'.\n"
        "2. Do NOT split the instruction text at commas, conjunctions, or phrase "
        "boundaries. Grammatical fragments of an instruction ('in between the edge', "
        "'the center') are NOT valid sub-steps.\n"
        "3. Only output MULTIPLE steps when the instruction describes an unambiguous "
        "sequence of physically distinct intermediate states the agent must pass "
        "through in order — e.g. 'tack then reach the far buoy' would become two steps. "
        "If in doubt, output a SINGLE step.\n"
        "4. The first step must describe an observable state reachable early in an episode.\n"
        "5. Every step must use vocabulary drawn from the observed language states above.\n"
        "6. No duplicate or overlapping steps. Maximum 5 steps.\n\n"
        "Output ONLY a numbered list, one step per line, no explanation."
    )
