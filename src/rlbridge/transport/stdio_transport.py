"""
Stdio Transport
================
Reads newline-delimited JSON-RPC requests from stdin and writes responses to
stdout.  Errors/logging go to stderr.

This transport is intended for use as a subprocess protocol (similar to how
MCP works with stdio), allowing any process to communicate with rlbridge without
needing an HTTP server.

Usage
-----
    python -m rlbridge.transport.stdio_transport
    # or
    from rlbridge.transport.stdio_transport import run_stdio_server
    run_stdio_server()
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TextIO

from ..environments.registry import registry
from ..server.dispatcher import rlbridgeDispatcher

log = logging.getLogger(__name__)


def run_stdio_server(
    dispatcher: rlbridgeDispatcher | None = None,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
) -> None:
    """
    Block forever reading newline-delimited JSON-RPC from *stdin* and writing
    responses to *stdout*.  Log to stderr.
    """
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    disp = dispatcher or rlbridgeDispatcher(registry=registry)

    for raw_line in stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            request_dict = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            _write(stdout, response)
            continue

        response = disp.dispatch(request_dict)
        _write(stdout, response)


def _write(stdout: TextIO, obj: dict) -> None:
    """Serialize *obj* as a single JSON line and flush."""
    try:
        line = json.dumps(obj, separators=(",", ":"))
        stdout.write(line + "\n")
        stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    run_stdio_server()
