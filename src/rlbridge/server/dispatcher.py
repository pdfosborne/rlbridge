"""
rlbridge Core Dispatcher
======================
Handles every JSON-RPC method call.  Transport-agnostic: accepts a dict
(already JSON-decoded) and returns a dict (ready for JSON encoding).

Both the FastAPI HTTP server and the stdio transport use this dispatcher.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..__init__ import __version__
from ..environments.registry import EnvironmentRegistry
from ..protocol.constants import ErrorCodes, Methods, rlbridge_PROTOCOL_VERSION
from ..protocol.messages import (
    CloseParams,
    CloseResult,
    CreateEnvironmentParams,
    CreateEnvironmentResult,
    InitializeParams,
    InitializeResult,
    ListEnvironmentsParams,
    ListEnvironmentsResult,
    ListInstancesResult,
    RenderParams,
    RenderResult,
    ResetParams,
    ResetResult,
    RpcError,
    RpcResponse,
    SpacesParams,
    SpacesResult,
    StepParams,
    StepResult,
)
from .exceptions import rlbridgeError
from .session import SessionManager

log = logging.getLogger(__name__)


class rlbridgeDispatcher:
    """Stateful JSON-RPC dispatcher for the rlbridge protocol."""

    def __init__(
        self,
        registry: EnvironmentRegistry,
        session: Optional[SessionManager] = None,
        max_instances: int = 64,
    ) -> None:
        self._registry = registry
        self._session = session or SessionManager(max_instances=max_instances)

    # ── Public entry-point ────────────────────────────────────────────────────

    def dispatch(self, request_dict: dict[str, Any]) -> dict[str, Any]:
        """Process one JSON-RPC request and return a JSON-RPC response dict."""
        req_id = request_dict.get("id", "null")

        # Basic validation
        if request_dict.get("jsonrpc") != "2.0":
            return self._error_response(
                req_id,
                ErrorCodes.INVALID_REQUEST,
                "jsonrpc field must be '2.0'",
            )

        method = request_dict.get("method")
        if not isinstance(method, str):
            return self._error_response(
                req_id,
                ErrorCodes.INVALID_REQUEST,
                "method must be a string",
            )

        params: dict[str, Any] = request_dict.get("params") or {}

        try:
            result = self._route(method, params)
            return RpcResponse(id=req_id, result=result).model_dump(exclude_none=False)
        except rlbridgeError as exc:
            log.warning("rlbridge application error: %s", exc)
            return self._error_response(req_id, exc.code, exc.message, exc.data)
        except Exception as exc:
            log.exception("Unhandled error in rlbridge dispatcher")
            return self._error_response(
                req_id,
                ErrorCodes.INTERNAL_ERROR,
                f"Internal server error: {exc}",
            )

    # ── Router ────────────────────────────────────────────────────────────────

    def _route(self, method: str, params: dict[str, Any]) -> Any:
        routes = {
            Methods.INITIALIZE:         self._handle_initialize,
            Methods.LIST_ENVIRONMENTS:  self._handle_list_environments,
            Methods.CREATE_ENVIRONMENT: self._handle_create_environment,
            Methods.RESET:              self._handle_reset,
            Methods.STEP:               self._handle_step,
            Methods.GET_SPACES:         self._handle_get_spaces,
            Methods.RENDER:             self._handle_render,
            Methods.CLOSE:              self._handle_close,
            Methods.LIST_INSTANCES:     self._handle_list_instances,
        }
        handler = routes.get(method)
        if handler is None:
            raise rlbridgeError(
                code=ErrorCodes.INVALID_REQUEST,
                message=f"Unknown method '{method}'",
                data={"method": method, "available": list(routes.keys())},
            )
        return handler(params)

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        p = InitializeParams.model_validate(params)
        result = InitializeResult(
            server_name="RL Bridge Server",
            server_version=__version__,
            protocol_version=rlbridge_PROTOCOL_VERSION,
            capabilities={
                "environments": True,
                "rendering": True,
                "sampling": True,
            },
        )
        log.info("Client '%s' %s connected.", p.client_name, p.client_version)
        return result.model_dump()

    def _handle_list_environments(self, params: dict[str, Any]) -> dict[str, Any]:
        p = ListEnvironmentsParams.model_validate(params)
        envs = self._registry.list_environments(
            tags=p.tags or None,
            namespace=p.namespace,
        )
        return ListEnvironmentsResult(environments=envs, total=len(envs)).model_dump()

    def _handle_create_environment(self, params: dict[str, Any]) -> dict[str, Any]:
        p = CreateEnvironmentParams.model_validate(params)
        try:
            env = self._registry.create(
                p.env_id,
                render_mode=p.render_mode,
                **p.kwargs,
            )
        except rlbridgeError:
            raise
        except Exception as exc:
            raise rlbridgeError(
                code=ErrorCodes.ENV_CREATION_FAILED,
                message=f"Failed to create '{p.env_id}': {exc}",
                data={"env_id": p.env_id},
            ) from exc

        record = self._session.create_instance(
            env_id=p.env_id,
            environment=env,
            render_mode=p.render_mode,
        )
        return CreateEnvironmentResult(
            instance_id=record.instance_id,
            env_id=p.env_id,
            observation_space=env.observation_space,
            action_space=env.action_space,
        ).model_dump()

    def _handle_reset(self, params: dict[str, Any]) -> dict[str, Any]:
        p = ResetParams.model_validate(params)
        record = self._session.get_instance(p.instance_id)
        result: ResetResult = record.environment.reset(
            seed=p.seed,
            options=p.options,
        )
        return result.model_dump()

    def _handle_step(self, params: dict[str, Any]) -> dict[str, Any]:
        p = StepParams.model_validate(params)
        record = self._session.get_instance(p.instance_id)
        result: StepResult = record.environment.step(p.action)
        return result.model_dump()

    def _handle_get_spaces(self, params: dict[str, Any]) -> dict[str, Any]:
        p = SpacesParams.model_validate(params)
        record = self._session.get_instance(p.instance_id)
        return SpacesResult(
            observation_space=record.environment.observation_space,
            action_space=record.environment.action_space,
        ).model_dump()

    def _handle_render(self, params: dict[str, Any]) -> dict[str, Any]:
        p = RenderParams.model_validate(params)
        record = self._session.get_instance(p.instance_id)
        try:
            result: RenderResult = record.environment.render()
        except NotImplementedError as exc:
            raise rlbridgeError(
                code=ErrorCodes.RENDER_UNAVAILABLE,
                message=str(exc),
            ) from exc
        return result.model_dump()

    def _handle_close(self, params: dict[str, Any]) -> dict[str, Any]:
        p = CloseParams.model_validate(params)
        closed = self._session.close_instance(p.instance_id)
        return CloseResult(closed=closed, instance_id=p.instance_id).model_dump()

    def _handle_list_instances(self, _params: dict[str, Any]) -> dict[str, Any]:
        instances = self._session.list_instances()
        return ListInstancesResult(instances=instances, total=len(instances)).model_dump()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _error_response(
        req_id: Any,
        code: int,
        message: str,
        data: Any = None,
    ) -> dict[str, Any]:
        error = RpcError(code=code, message=message, data=data)
        return RpcResponse(
            id=str(req_id),
            error=error,
        ).model_dump(exclude_none=False)
