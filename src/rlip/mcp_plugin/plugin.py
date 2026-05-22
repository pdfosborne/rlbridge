"""
RLIP MCP Plugin for Claude Code
=================================
Exposes RL environments as MCP tools so Claude Code can interact with
Gymnasium environments directly via natural-language agent tasks.

How it works
------------
1. This file is the entry-point for the MCP server using FastMCP (stdio
   transport).  All tool and resource definitions live in the _tools_*.py
   submodules; importing them here registers every tool with the shared
   ``mcp`` FastMCP instance created in ``_state``.
2. Claude Code launches it as a subprocess and communicates over stdin/stdout.
3. The plugin can run environments *in-process* (default) or *proxy* to a
   running RLIP HTTP server (set RLIP_SERVER_URL env var).

Configuration
-------------
Run the matching install command once, then restart the client:

Claude Code  (~/.claude.json):
    rlip install-claude
    # or manually:
    {
      "mcpServers": {
        "rlip": {
          "command": "python",
          "args": ["-m", "rlip.mcp_plugin"],
          "type": "stdio"
        }
      }
    }

Claude Desktop (GUI – macOS/Windows/Linux):
    rlip install-claude-desktop
    # or manually add to claude_desktop_config.json:
    {
      "mcpServers": {
        "rlip": {
          "command": "python",
          "args": ["-m", "rlip.mcp_plugin"]
        }
      }
    }

LM Studio  (~/.lmstudio/mcp.json or platform equivalent):
    rlip install-lmstudio
    # or manually add to mcp.json:
    {
      "mcpServers": {
        "rlip": {
          "command": "python",
          "args": ["-m", "rlip.mcp_plugin"],
          "type": "stdio"
        }
      }
    }

Cursor  (~/.cursor/mcp.json):
    rlip install-cursor
    # or manually:
    {
      "mcpServers": {
        "rlip": {
          "command": "python",
          "args": ["-m", "rlip.mcp_plugin"]
        }
      }
    }

Windsurf  (~/.codeium/windsurf/mcp_config.json):
    rlip install-windsurf
    # or manually:
    {
      "mcpServers": {
        "rlip": {
          "command": "python",
          "args": ["-m", "rlip.mcp_plugin"]
        }
      }
    }

Codex CLI  (~/.codex/config.toml):
    rlip install-codex
    # or manually:
    [mcp_servers.rlip]
    command = "python"
    args = ["-m", "rlip.mcp_plugin"]

OpenCode  (~/.config/opencode/config.json):
    rlip install-opencode
    # or manually:
    {
      "mcp": {
        "rlip": {
          "type": "local",
          "command": ["python", "-m", "rlip.mcp_plugin"]
        }
      }
    }

If installed via pip as a console script, replace ``["python", "-m",
"rlip.mcp_plugin"]`` with ``"rlip-mcp"`` (use ``--use-script`` flag).

Module layout
-------------
_state.py            – FastMCP instance + all shared mutable caches
_dispatch.py         – _dispatch(), _proxy_dispatch(), _fmt_obs()
_env_wrappers.py     – _ShapedEnv, _LangStateEnv
_tools_env.py        – basic environment interaction tools
_tools_instruction.py– instruction-following tools
_tools_instruction_plan.py – instruction plan DB and cache tools (rl_get_instruction_plan, rl_list_cached_instructions, rl_clear_instruction_cache)
_tools_agents_training.py  – RL agent training tools (rl_train_agent, rl_get_training_result)
_tools_agents_utils.py     – RL agent utilities (rl_list_agents) and training helpers
_tools_agents_io.py        – RL agent save/load (rl_list_trained_agents, rl_load_agent)
_tools_agents_eval.py      – RL agent evaluation (rl_run_agent_episode, rl_create_training_report, rl_evaluate_agent)
_tools_experiment.py       – Automated RL experiment pipeline (rl_experiment_process)
_tools_saved_experiments.py– Saved experiment management (rl_save_experiment, rl_list_experiments, rl_load_experiment)
_tools_builder.py    – custom environment builder tools (wrap Gymnasium envs)
_tools_cache.py      – environment cache tools (rl_load_cached_environments, rl_list_cached_environments, rl_clear_obs_cache)
_tools_create_env.py – custom environment creation from code (rl_create_environment_from_code, rl_get_environment_template, rl_validate_environment_code, rl_delete_custom_environment)
_tools_render.py     – rendering tools and MCP resources
_tools_dashboard.py  – live training dashboard server tools
env custom tools     – discovered from environment modules at startup
plugin.py            – this file: entry-point + main()
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

from ._state import _in_process, _registry, log, mcp

# Importing these submodules has the side-effect of decorating every tool
# function with @mcp.tool() / @mcp.resource(), registering them with the
# shared FastMCP instance created in _state.
from . import (  # noqa: F401, E402
    _tools_env,
    _tools_instruction,
    _tools_instruction_plan,
    _tools_agents_training,
    _tools_agents_utils,
    _tools_agents_eval,
    _tools_agents_io,
    _tools_experiment,
    _tools_llm_play,
    _tools_builder,
    _tools_cache,
    _tools_create_env,
    _tools_render,
    _tools_derived_instructions,
    _tools_dashboard,
    _tools_saved_experiments,
)


def _register_environment_custom_tools() -> int:
    """Discover and register custom tools exported by environment modules.

    Built-in predefined modules may define
    ``register_mcp_tools(mcp=..., registry=..., log=...)``.  Installed
    third-party packages register the same hook via the
    ``rlip.environment_mcp_tools`` entry-point group.
    """
    total_registered = 0
    try:
        from ..environments import predefined as predefined_pkg
    except Exception as exc:
        log.warning("Unable to import predefined environments package: %s", exc)
        predefined_pkg = None

    if predefined_pkg is not None:
        module_prefix = predefined_pkg.__name__ + "."
        for mod in pkgutil.iter_modules(predefined_pkg.__path__, prefix=module_prefix):
            try:
                module = importlib.import_module(mod.name)
            except Exception as exc:
                log.debug("Skipping env module %s (import failed): %s", mod.name, exc)
                continue

            register = getattr(module, "register_mcp_tools", None)
            if not callable(register):
                continue

            try:
                added = int(register(mcp=mcp, registry=_registry, log=log) or 0)
                total_registered += max(0, added)
                if added:
                    log.info("Registered %d custom MCP tools from %s", added, mod.name)
            except Exception as exc:
                log.warning("register_mcp_tools failed for %s: %s", mod.name, exc)

    if _in_process and _registry is not None:
        try:
            from ..environments.plugins import load_plugin_mcp_tools
            from ._state import _trained_agents

            total_registered += load_plugin_mcp_tools(
                mcp=mcp,
                registry=_registry,
                log=log,
                trained_agents=_trained_agents,
            )
        except Exception as exc:
            log.warning("load_plugin_mcp_tools failed: %s", exc)

    return total_registered


# ── Entry-point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Entry-point called by the ``rlip-mcp`` console script."""
    logging.basicConfig(level=logging.WARNING)
    _register_environment_custom_tools()
    # Auto-register all Gymnasium environments that are installed
    if _in_process:
        try:
            count = _registry.auto_register_gymnasium()
            log.info("Auto-registered %d Gymnasium environments.", count)
        except Exception as exc:
            log.warning("auto_register_gymnasium failed: %s", exc)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
