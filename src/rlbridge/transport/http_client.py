"""
rlbridge HTTP Client
=================
Synchronous and async clients for communicating with a remote rlbridge HTTP server.

Synchronous example
-------------------
    from rlbridge.transport.http_client import rlbridgeClient

    client = rlbridgeClient("http://localhost:8765")
    client.initialize()
    envs = client.list_environments()
    instance_id = client.create_environment("CartPole-v1")
    obs, info = client.reset(instance_id)
    step_result = client.step(instance_id, action=1)
    client.close_environment(instance_id)
    client.close()

Async example
-------------
    async with AsyncrlbridgeClient("http://localhost:8765") as client:
        await client.initialize()
        envs = await client.list_environments()
        ...
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from ..protocol.constants import Methods
from ..protocol.messages import (
    CreateEnvironmentResult,
    InitializeResult,
    ListEnvironmentsResult,
    ListInstancesResult,
    RenderResult,
    ResetResult,
    SpacesResult,
    StepResult,
    CloseResult,
)


# ── Shared RPC helpers ────────────────────────────────────────────────────────

def _make_request(method: str, params: dict[str, Any]) -> dict[str, Any]:
    import uuid
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params,
    }


def _check_response(response: dict[str, Any]) -> Any:
    if "error" in response and response["error"] is not None:
        err = response["error"]
        raise rlbridgeClientError(
            code=err.get("code", -1),
            message=err.get("message", "Unknown error"),
            data=err.get("data"),
        )
    return response.get("result")


class rlbridgeClientError(Exception):
    """Raised when the rlbridge server returns an error response."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.data = data


# ── Synchronous client ────────────────────────────────────────────────────────

class rlbridgeClient:
    """
    Blocking rlbridge client backed by ``httpx``.
    """

    def __init__(self, base_url: str = "http://localhost:8765", timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self._base_url, timeout=timeout)

    def _rpc(self, method: str, **params: Any) -> Any:
        body = _make_request(method, params)
        resp = self._http.post("/rpc", json=body)
        resp.raise_for_status()
        return _check_response(resp.json())

    # ── Protocol methods ──────────────────────────────────────────────────────

    def initialize(
        self,
        client_name: str = "rlbridge-python-client",
        client_version: str = "0.1.0",
    ) -> InitializeResult:
        result = self._rpc(
            Methods.INITIALIZE,
            client_name=client_name,
            client_version=client_version,
        )
        return InitializeResult.model_validate(result)

    def list_environments(
        self,
        tags: Optional[list[str]] = None,
        namespace: Optional[str] = None,
    ) -> ListEnvironmentsResult:
        result = self._rpc(
            Methods.LIST_ENVIRONMENTS,
            tags=tags or [],
            namespace=namespace,
        )
        return ListEnvironmentsResult.model_validate(result)

    def create_environment(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> CreateEnvironmentResult:
        result = self._rpc(
            Methods.CREATE_ENVIRONMENT,
            env_id=env_id,
            render_mode=render_mode,
            kwargs=kwargs,
        )
        return CreateEnvironmentResult.model_validate(result)

    def reset(
        self,
        instance_id: str,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        result = self._rpc(
            Methods.RESET,
            instance_id=instance_id,
            seed=seed,
            options=options or {},
        )
        return ResetResult.model_validate(result)

    def step(self, instance_id: str, action: Any) -> StepResult:
        result = self._rpc(Methods.STEP, instance_id=instance_id, action=action)
        return StepResult.model_validate(result)

    def get_spaces(self, instance_id: str) -> SpacesResult:
        result = self._rpc(Methods.GET_SPACES, instance_id=instance_id)
        return SpacesResult.model_validate(result)

    def render(self, instance_id: str) -> RenderResult:
        result = self._rpc(Methods.RENDER, instance_id=instance_id)
        return RenderResult.model_validate(result)

    def close_environment(self, instance_id: str) -> CloseResult:
        result = self._rpc(Methods.CLOSE, instance_id=instance_id)
        return CloseResult.model_validate(result)

    def list_instances(self) -> ListInstancesResult:
        result = self._rpc(Methods.LIST_INSTANCES)
        return ListInstancesResult.model_validate(result)

    # ── Context manager ───────────────────────────────────────────────────────

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "rlbridgeClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ── Async client ──────────────────────────────────────────────────────────────

class AsyncrlbridgeClient:
    """
    Async rlbridge client backed by ``httpx.AsyncClient``.
    """

    def __init__(self, base_url: str = "http://localhost:8765", timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def _rpc(self, method: str, **params: Any) -> Any:
        body = _make_request(method, params)
        resp = await self._http.post("/rpc", json=body)
        resp.raise_for_status()
        return _check_response(resp.json())

    async def initialize(
        self,
        client_name: str = "rlbridge-async-client",
        client_version: str = "0.1.0",
    ) -> InitializeResult:
        result = await self._rpc(
            Methods.INITIALIZE,
            client_name=client_name,
            client_version=client_version,
        )
        return InitializeResult.model_validate(result)

    async def list_environments(
        self,
        tags: Optional[list[str]] = None,
        namespace: Optional[str] = None,
    ) -> ListEnvironmentsResult:
        result = await self._rpc(
            Methods.LIST_ENVIRONMENTS,
            tags=tags or [],
            namespace=namespace,
        )
        return ListEnvironmentsResult.model_validate(result)

    async def create_environment(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> CreateEnvironmentResult:
        result = await self._rpc(
            Methods.CREATE_ENVIRONMENT,
            env_id=env_id,
            render_mode=render_mode,
            kwargs=kwargs,
        )
        return CreateEnvironmentResult.model_validate(result)

    async def reset(
        self,
        instance_id: str,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> ResetResult:
        result = await self._rpc(
            Methods.RESET,
            instance_id=instance_id,
            seed=seed,
            options=options or {},
        )
        return ResetResult.model_validate(result)

    async def step(self, instance_id: str, action: Any) -> StepResult:
        result = await self._rpc(Methods.STEP, instance_id=instance_id, action=action)
        return StepResult.model_validate(result)

    async def get_spaces(self, instance_id: str) -> SpacesResult:
        result = await self._rpc(Methods.GET_SPACES, instance_id=instance_id)
        return SpacesResult.model_validate(result)

    async def render(self, instance_id: str) -> RenderResult:
        result = await self._rpc(Methods.RENDER, instance_id=instance_id)
        return RenderResult.model_validate(result)

    async def close_environment(self, instance_id: str) -> CloseResult:
        result = await self._rpc(Methods.CLOSE, instance_id=instance_id)
        return CloseResult.model_validate(result)

    async def list_instances(self) -> ListInstancesResult:
        result = await self._rpc(Methods.LIST_INSTANCES)
        return ListInstancesResult.model_validate(result)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncrlbridgeClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
