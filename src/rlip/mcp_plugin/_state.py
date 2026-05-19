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

from ._prompts import rlip_system_prompt

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
    instructions=rlip_system_prompt(),
)


# ── Shared session caches ─────────────────────────────────────────────────────

# InstructionFollowingProtocol instances keyed by match_id.
# Shared between _tools_instruction (writes) and _tools_agents (reads).
_instruction_protocols: dict[str, Any] = {}

# Trained agent entries keyed by agent_id.
# Shared between _tools_agents (writes) and _tools_render (reads).
_trained_agents: dict[str, Any] = {}

# Background training jobs keyed by job_id.
# Written by rl_train_agent / rl_train_and_derive_instructions;
# read by rl_get_training_result.
_training_jobs: dict[str, Any] = {}

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


def _env_agents_registry_path(env_id: str) -> Path:
    """Path to the persistent trained-agent registry for *env_id*.

    Keyed by agent_id; stores instruction, eval_reward, artifact path, etc.
    Saved in the cache directory so it survives across working directories.
    """
    return _cache_root() / "environments" / _safe_env_name(env_id) / "agents_registry.json"


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
