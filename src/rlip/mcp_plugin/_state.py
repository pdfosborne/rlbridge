"""
Shared state for the RLIP MCP plugin.

This module is imported first by every ``_tools_*.py`` submodule.  It:

- Selects in-process vs. proxy mode based on ``RLIP_SERVER_URL``.
- Creates the single ``FastMCP`` instance (``mcp``) that all tool submodules
  decorate their functions with.
- Holds all mutable session caches (``_instruction_protocols``,
  ``_trained_agents``, ``_custom_translators``, ``_sampled_states``) so that
  tools defined in different submodules share the same live dictionaries.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)

# ── Select in-process vs proxy mode ──────────────────────────────────────────

_RLIP_SERVER_URL = os.environ.get("RLIP_SERVER_URL", "")

if _RLIP_SERVER_URL:
    # Proxy mode – forward calls to a running RLIP HTTP server
    from ..transport.http_client import RLIPClient as _RLIPClient
    _proxy: Any = _RLIPClient(_RLIP_SERVER_URL)
    _in_process = False
    _registry: Any = None
    _dispatcher: Any = None
else:
    # In-process mode – run environments directly in this process
    from ..environments.registry import registry as _registry  # type: ignore[assignment]
    from ..server.dispatcher import RLIPDispatcher as _Dispatcher
    from ..server.session import SessionManager as _SessionManager
    _session = _SessionManager(max_instances=int(os.environ.get("RLIP_MAX_INSTANCES", "16")))
    _dispatcher = _Dispatcher(registry=_registry, session=_session)
    _in_process = True
    _proxy = None


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
        "4. rl_render_policy(env_id, agent_id=agent_id) – render the best training episode as a GIF.\n"
        "5. rl_create_training_report(agent_id, compare_agent_ids=[...]) – generate "
        "a comparative PNG report showing reward convergence, optimal-policy reward "
        "at breakpoints, instruction-match details, and training metadata/hyper-parameters.\n\n"
        "IMPORTANT — comparing multiple agents: when training more than one agent "
        "(e.g. tabular_q vs dqn vs ppo, or with/without sub-goal shaping), always "
        "use identical values for n_episodes, max_steps, seed, gamma, and any other "
        "shared hyper-parameter across every rl_train_agent call.  Only vary the "
        "parameter(s) under investigation.  This ensures that differences in "
        "reported reward are attributable to the agent or configuration being "
        "tested, not to unequal training budgets or random seeds.\n\n"
        "Combined instruction-following + agent training workflow:\n"
        "1. rl_match_instruction(env_id, instruction) – explore and find the sub-goal state. "
        "Returns a match_id.\n"
        "2. rl_train_agent(agent_type, env_id, match_id=match_id, use_language_state=True) – "
        "train with sub-goal reward shaping AND language observations (use_language_state is "
        "optional but recommended when a translator is available).\n"
        "3. rl_render_policy(env_id, agent_id=agent_id) – render the result.\n"
        "4. rl_create_training_report(agent_id, compare_agent_ids=[...]) – comparative "
        "training report including instruction match observation and similarity percentage.\n\n"
        "Auto-derived instruction workflow (no instruction needed up front):\n"
        "1. rl_train_and_derive_instructions(env_id, agent_type, n_episodes) – train an "
        "agent while tracking which language states appear in successful episodes.  "
        "Automatically scores and caches the top-k instruction candidates.\n"
        "2. rl_list_cached_instructions(env_id) – inspect cached instructions with "
        "per-instruction success-rate statistics.\n"
        "3. rl_apply_derived_instruction(env_id, instruction) – convert a cached "
        "instruction into a match_id (no re-exploration needed).\n"
        "4. rl_instruction_run_episode(match_id) or rl_train_agent(match_id=...) – "
        "run sub-goal-shaped episodes using the derived instruction."
    ),
)


# ── Shared session caches ─────────────────────────────────────────────────────

# InstructionFollowingProtocol instances keyed by match_id.
# Shared between _tools_instruction (writes) and _tools_agents (reads).
_instruction_protocols: dict[str, Any] = {}

# Trained agent entries keyed by agent_id.
# Shared between _tools_agents (writes) and _tools_render (reads).
_trained_agents: dict[str, Any] = {}

# Active custom language translators registered in this session.
# Shared between _tools_builder (writes) and _tools_agents/_tools_render (reads).
_custom_translators: dict[str, Any] = {}

# Sampled states cache — helps rl_set_translator_code show test translations
# without requiring the caller to pass the states back.
_sampled_states: dict[str, list[Any]] = {}

def _safe_env_name(env_id: str) -> str:
    """Normalize env IDs for filesystem-safe directory names."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", env_id).strip("._-")
    return cleaned or "environment"


# All plugin artifacts live under the working directory Claude was launched in.
# Users can override this with RLIP_OUTPUT_ROOT if desired.
_OUTPUT_ROOT = Path(os.environ.get("RLIP_OUTPUT_ROOT") or Path.cwd())
_RLIP_ROOT = _OUTPUT_ROOT / ".rlip"

# Shared roots
_RENDERS_DIR = _RLIP_ROOT / "renders"
_CUSTOM_ENV_CACHE_ROOT = _RLIP_ROOT / "envs"
_CATALOG_PATH = _RLIP_ROOT / "catalog.json"
_ENV_ARTIFACTS_ROOT = _RLIP_ROOT / "environments"


def _env_root_dir(env_id: str) -> Path:
    """Per-environment root for renders, cache, and agent artifacts."""
    return _ENV_ARTIFACTS_ROOT / _safe_env_name(env_id)


def _env_renders_dir(env_id: str) -> Path:
    return _env_root_dir(env_id) / "renders"


def _env_cache_dir(env_id: str) -> Path:
    return _env_root_dir(env_id) / "cache"


def _env_agents_dir(env_id: str) -> Path:
    return _env_root_dir(env_id) / "agents"
