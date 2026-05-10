"""Adapters sub-package — connects RLIP to non-MCP frontends."""

from .openai_agent import RLIPAgent, RLIP_TOOLS

__all__ = ["RLIPAgent", "RLIP_TOOLS"]
