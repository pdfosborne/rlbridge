"""Adapters sub-package - connects rlbridge to non-MCP frontends."""

from .openai_agent import rlbridgeAgent, rlbridge_TOOLS

__all__ = ["rlbridgeAgent", "rlbridge_TOOLS"]
