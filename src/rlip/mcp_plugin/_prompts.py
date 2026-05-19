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
    rl_list_agents() and validated in rl_train_agent / rl_train_agent_auto.

Call-site labels
----------------
TOOL: rl_match_instruction  (also used by rl_match_sequential_instructions)
    decompose_instruction_vocab_prompt()

TOOL: rl_train_and_derive_instructions
    decompose_instruction_simple_prompt()
"""

from __future__ import annotations


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
# TOOL: rl_match_instruction
# TOOL: rl_match_sequential_instructions
# ---------------------------------------------------------------------------

def decompose_instruction_vocab_prompt(
    env_id: str,
    instruction: str,
    obs_block: str,
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
    obs_block:
        Pre-formatted block of observed state descriptions (one per line,
        indented with ``  - ``).  Capped at ~80 entries by the caller.
    vocab_block:
        Pre-formatted block of recurring vocabulary clauses extracted from
        *obs_block* (one per line, indented with ``  • ``).  Capped at ~60
        entries by the caller.
    """
    return (
        f"You are setting up reward shaping for a reinforcement learning agent "
        f"in the environment \"{env_id}\".\n\n"
        f"The user's instruction is:\n  \"{instruction}\"\n\n"
        f"These are ALL unique state descriptions the environment's language "
        f"translator can produce — they are the ONLY valid vocabulary:\n"
        f"{obs_block}\n\n"
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
    obs_block: str,
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
    obs_block:
        Pre-formatted block of observed language state strings (one per line,
        indented with ``  - ``).  Capped at ~60 entries by the caller.
    """
    return (
        f"You are helping set up sequential reward shaping for RL in environment '{env_id}'.\n\n"
        f"High-level instruction:\n  '{instruction}'\n\n"
        f"Observed environment language states:\n{obs_block}\n\n"
        "Break the instruction into 2-5 ordered, distinct, concrete sub-steps. "
        "The FIRST step must focus on what to do at episode start. "
        "Use environment vocabulary. Avoid duplicate or overlapping steps. "
        "Each sub-step must describe an observable environment state, position, orientation, transition, "
        "or condition that could be matched directly to the environment's language translations. "
        "Do not leave instructions as abstract action-only jargon. If the user gives an abstract maneuver, "
        "rewrite it as successive observable intermediate states. For example, a sailing maneuver like tack "
        "should become turning-state instructions that describe the boat's turn in observable stages. "
        "ALL instructions should be written to match the problem context. "
        "ALL instructions must use language that aligns with the environment's observed language translations. "
        "Output ONLY a numbered list, one step per line."
    )
