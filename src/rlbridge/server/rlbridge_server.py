"""Backward-compatible server module path.

Historically, server code was imported from ``rlbridge.server.rlbridge_server``.
Keep that import path valid by re-exporting ``create_app`` from the current
implementation module.
"""

from .rlip_server import create_app

__all__ = ["create_app"]
