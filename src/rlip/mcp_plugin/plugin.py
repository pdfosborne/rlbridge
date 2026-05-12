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

Claude Code configuration  (~/.claude.json)
-------------------------------------------
    {
      "mcpServers": {
        "rlip": {
          "command": "python",
          "args": ["-m", "rlip.mcp_plugin"],
          "type": "stdio"
        }
      }
    }

Or if you installed via pip as a script:
    {
      "mcpServers": {
        "rlip": {
          "command": "rlip-mcp",
          "type": "stdio"
        }
      }
    }

Module layout
-------------
_state.py            – FastMCP instance + all shared mutable caches
_dispatch.py         – _dispatch(), _proxy_dispatch(), _fmt_obs()
_env_wrappers.py     – _ShapedEnv, _LangStateEnv
_tools_env.py        – basic environment interaction tools
_tools_instruction.py– instruction-following tools
_tools_agents.py     – RL agent training / evaluation tools
_tools_builder.py    – custom environment builder tools
_tools_render.py     – rendering tools and MCP resources
_tools_dashboard.py  – live training dashboard server tools
plugin.py            – this file: entry-point + main()
"""

from __future__ import annotations

import logging

from ._state import _in_process, _registry, log, mcp

# Importing these submodules has the side-effect of decorating every tool
# function with @mcp.tool() / @mcp.resource(), registering them with the
# shared FastMCP instance created in _state.
from . import (  # noqa: F401, E402
    _tools_env,
    _tools_instruction,
    _tools_agents,
    _tools_builder,
    _tools_render,
    _tools_derived_instructions,
    _tools_dashboard,
)


# ── Entry-point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Entry-point called by the ``rlip-mcp`` console script."""
    logging.basicConfig(level=logging.WARNING)
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
