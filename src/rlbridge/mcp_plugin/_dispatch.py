"""
Dispatch helpers for the rlbridge MCP plugin.

Routes method calls to either the in-process RLIPDispatcher or the remote
proxy (RLIPClient) depending on the mode selected in ``_state``.
"""

from __future__ import annotations

import json
from typing import Any

from ._state import _dispatcher, _in_process, _proxy


def _dispatch(method: str, **params: Any) -> Any:
    """Route a call to either the in-process dispatcher or remote proxy."""
    if _in_process:
        request = {"jsonrpc": "2.0", "id": "mcp", "method": method, "params": params}
        response = _dispatcher.dispatch(request)
        if response.get("error"):
            err = response["error"]
            raise RuntimeError(f"[{err['code']}] {err['message']}")
        return response.get("result", {})
    else:
        return _proxy_dispatch(method, params)


def _proxy_dispatch(method: str, params: dict[str, Any]) -> Any:
    """Dispatch to the remote HTTP proxy client."""
    from ..protocol.constants import Methods
    mapping = {
        Methods.INITIALIZE:         lambda p: _proxy.initialize(**p),
        Methods.LIST_ENVIRONMENTS:  lambda p: _proxy.list_environments(**p).model_dump(),
        Methods.CREATE_ENVIRONMENT: lambda p: _proxy.create_environment(**p).model_dump(),
        Methods.RESET:              lambda p: _proxy.reset(**p).model_dump(),
        Methods.STEP:               lambda p: _proxy.step(**p).model_dump(),
        Methods.GET_SPACES:         lambda p: _proxy.get_spaces(**p).model_dump(),
        Methods.RENDER:             lambda p: _proxy.render(**p).model_dump(),
        Methods.CLOSE:              lambda p: _proxy.close_environment(**p).model_dump(),
        Methods.LIST_INSTANCES:     lambda p: _proxy.list_instances().model_dump(),
    }
    fn = mapping.get(method)
    if fn is None:
        raise ValueError(f"Unknown method: {method}")
    return fn(params)


def _fmt_obs(obs: Any) -> str:
    """Format an observation for readable MCP tool output."""
    if isinstance(obs, list):
        if len(obs) <= 16:
            formatted = ", ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in obs)
            return f"[{formatted}]"
        return f"<array of {len(obs)} values, first 8: {obs[:8]}>"
    return json.dumps(obs, indent=2)
