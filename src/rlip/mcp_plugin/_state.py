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
        "Storage layout:\n"
        "  • ~/.rlip/  (cache) – custom env definitions, catalog, language-translation "
        "source, and instruction data.  Managed automatically; no need to touch.\n"
        "  • <cwd>/rlip_results/  (local saves) – policy render GIFs, training-report "
        "PNGs, and trained-agent ZIP packages.  Everything here is yours to keep.\n\n"
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
        "MANDATORY PLANNING RULE: You MUST call rl_get_instruction_plan(env_id) as the "
        "FIRST action whenever starting work on an environment or deciding what to try "
        "next.  Do NOT call rl_match_instruction() or rl_train_and_derive_instructions() "
        "before you have read the plan.  If the plan is empty, proceed normally.  "
        "If it has entries, use the eval rewards, derived scores, and matched states to "
        "choose the best next instruction rather than repeating something already tried.  "
        "Always explain your reasoning to the user based on the plan contents.\n\n"
        "Combined instruction-following + agent training workflow:\n"
        "0. rl_get_instruction_plan(env_id) – ALWAYS call this first.  Use the results "
        "to decide which instruction to try (or whether to derive first).\n"
        "1. rl_match_instruction(env_id, instruction) – explore and find the sub-goal state. "
        "Returns a match_id.\n"
        "2. rl_train_agent(agent_type, env_id, match_id=match_id, use_language_state=True) – "
        "train with sub-goal reward shaping AND language observations (use_language_state is "
        "optional but recommended when a translator is available).\n"
        "3. rl_render_policy(env_id, agent_id=agent_id) – render the result.\n"
        "4. rl_create_training_report(agent_id, compare_agent_ids=[...]) – comparative "
        "training report including instruction match observation and similarity percentage.\n\n"
        "Auto-derived instruction workflow (no instruction needed up front):\n"
        "0. rl_get_instruction_plan(env_id) – ALWAYS call this first to check if "
        "derived instructions already exist before running a fresh derivation.\n"
        "1. rl_train_and_derive_instructions(env_id, agent_type, n_episodes) – train an "
        "agent while tracking which language states appear in successful episodes.  "
        "Automatically scores and caches the top-k instruction candidates.\n"
        "2. rl_list_cached_instructions(env_id) – inspect cached instructions with "
        "per-instruction success-rate statistics.\n"
        "3. rl_apply_derived_instruction(env_id, instruction) – convert a cached "
        "instruction into a match_id (no re-exploration needed).\n"
        "4. rl_instruction_run_episode(match_id) or rl_train_agent(match_id=...) – "
        "run sub-goal-shaped episodes using the derived instruction.\n\n"
        "Instruction planning database:\n"
        "Every call to rl_match_instruction() and rl_train_and_derive_instructions() "
        "automatically records the instruction, its sub-steps, and outcome in a "
        "persistent per-environment database saved to "
        "~/.rlip/environments/<env>/instruction_plan.json.  "
        "After rl_train_agent() completes with a match_id, a clean evaluation episode "
        "(no instruction bonus) is run automatically and the eval_reward is stored.\n"
        "• rl_get_instruction_plan(env_id) – read the full planning database: shows "
        "every tried instruction, its source (llm/derived), best clean eval reward, "
        "derived score, similarity %, matched environment state, and usage history.  "
        "ALWAYS call this before any planning decision."
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


# ---------------------------------------------------------------------------
# Path resolution — two distinct roots
# ---------------------------------------------------------------------------
#
# CACHE ROOT  (~/.rlip  or  RLIP_CACHE_ROOT)
#   Used *internally* by the plugin to persist data that the agent tools
#   read back automatically: custom environment definitions, the env catalog,
#   language-translation source, instruction-matching data, and environment-
#   source copies.  This is a stable, user-home location so it survives across
#   different working directories.
#
# LOCAL OUTPUT ROOT  (<cwd>/rlip_results  or  RLIP_LOCAL_OUTPUT)
#   User-facing outputs written wherever Claude/Codex/etc. is run from.
#   Includes policy-render GIFs, training-report PNGs, and trained-agent ZIP
#   packages.  Structured as  rlip_results/<env_name>/{renders,reports,agents}/
#   so everything is easy to find in the project directory.
#
# Both roots are resolved *lazily* (at call-time) to handle clients that set
# the process cwd *after* import (e.g. Claude Desktop on Windows).
#
# Legacy env var RLIP_OUTPUT_ROOT is still honoured as an alias for
# RLIP_CACHE_ROOT so existing configs keep working.
# ---------------------------------------------------------------------------

def _cache_root() -> Path:
    """Return (and guarantee existence of) the RLIP cache directory.

    This is where the plugin stores data it reads back automatically:
    custom environments, the env catalog, language-translation source,
    and instruction-matching data.

    Resolution order:
      1. ``RLIP_CACHE_ROOT`` environment variable
      2. ``RLIP_OUTPUT_ROOT`` environment variable  (legacy alias)
      3. User home directory  (~/.rlip)
    """
    env_override = os.environ.get("RLIP_CACHE_ROOT") or os.environ.get("RLIP_OUTPUT_ROOT")
    if env_override:
        root = Path(env_override) / ".rlip"
    else:
        root = Path.home() / ".rlip"
    root.mkdir(parents=True, exist_ok=True)
    return root


# Keep legacy name as an alias so any external code still compiles.
_rlip_root = _cache_root


def _local_output_root() -> Path:
    """Return (and guarantee existence of) the local results directory.

    Outputs intended for the user (renders, reports, agent packages) are
    written here so they land in the directory where Claude/Codex is run from.

    Resolution order:
      1. ``RLIP_LOCAL_OUTPUT`` environment variable
      2. ``<cwd>/rlip_results``
    """
    env_override = os.environ.get("RLIP_LOCAL_OUTPUT")
    root = Path(env_override) if env_override else Path.cwd() / "rlip_results"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ── Cache-backed paths (internal; survives across cwds) ──────────────────────

def _env_cache_dir(env_id: str) -> Path:
    """Per-environment cache: translator source, env source, instruction data."""
    return _cache_root() / "environments" / _safe_env_name(env_id) / "cache"


def _env_plan_db_path(env_id: str) -> Path:
    """Path to the persistent instruction plan database for *env_id*.

    Stored in the cache directory (not the local results root) because it
    is metadata consumed by the LLM tool layer, not a user artefact.
    """
    return _cache_root() / "environments" / _safe_env_name(env_id) / "instruction_plan.json"


def _custom_env_cache_root() -> Path:
    return _cache_root() / "envs"


def _catalog_path() -> Path:
    return _cache_root() / "catalog.json"


# ── Local-output paths (user-facing; relative to cwd) ────────────────────────

def _local_env_dir(env_id: str) -> Path:
    """Per-environment directory inside the local results root."""
    return _local_output_root() / _safe_env_name(env_id)


def _env_renders_dir(env_id: str) -> Path:
    """Where policy-render GIFs are saved (local results directory)."""
    return _local_env_dir(env_id) / "renders"


def _env_agents_dir(env_id: str) -> Path:
    """Where trained-agent ZIP packages are saved (local results directory)."""
    return _local_env_dir(env_id) / "agents"


def _env_reports_dir(env_id: str) -> Path:
    """Where training-report PNGs are saved (local results directory)."""
    return _local_env_dir(env_id) / "reports"
