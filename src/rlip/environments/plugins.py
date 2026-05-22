"""
Third-party environment plugins
================================
Discover and register environments shipped as separate pip packages via
setuptools entry points:

- ``rlip.environments`` – register environment factories
- ``rlip.environment_mcp_tools`` – register environment-specific MCP tools
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

log = logging.getLogger(__name__)

_ENV_ENTRY_GROUP = "rlip.environments"
_MCP_ENTRY_GROUP = "rlip.environment_mcp_tools"


def load_plugin_environments(registry: Any = None) -> int:
    """Load and register all installed ``rlip.environments`` entry points."""
    if registry is None:
        from .registry import registry as default_registry

        registry = default_registry

    total = 0
    for ep in entry_points(group=_ENV_ENTRY_GROUP):
        try:
            register_fn = ep.load()
            count = int(register_fn(registry=registry) or 0)
            total += max(0, count)
            if count:
                log.info("Registered %d environment(s) from plugin %r", count, ep.name)
        except Exception as exc:
            log.warning("Failed to load environment plugin %r: %s", ep.name, exc)
    return total


def load_plugin_mcp_tools(
    *,
    mcp: Any,
    registry: Any,
    log: Any,
    **kwargs: Any,
) -> int:
    """Load and register all installed ``rlip.environment_mcp_tools`` entry points."""
    total = 0
    for ep in entry_points(group=_MCP_ENTRY_GROUP):
        try:
            register_fn = ep.load()
            added = int(
                register_fn(mcp=mcp, registry=registry, log=log, **kwargs) or 0
            )
            total += max(0, added)
            if added:
                log.info(
                    "Registered %d custom MCP tool(s) from plugin %r",
                    added,
                    ep.name,
                )
        except Exception as exc:
            log.warning("Failed to load MCP tools plugin %r: %s", ep.name, exc)
    return total
