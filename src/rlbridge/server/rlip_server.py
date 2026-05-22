"""
rlbridge HTTP Server (FastAPI)
===========================
Exposes the rlbridge protocol over HTTP as a JSON-RPC 2.0 endpoint.

    POST /rpc          - single JSON-RPC request
    POST /rpc/batch    - batch of JSON-RPC requests
    GET  /health       - liveness probe
    GET  /info         - server metadata
    GET  /environments - shorthand REST list of available environments
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..__init__ import __version__
from ..environments.registry import EnvironmentRegistry, registry as default_registry
from ..protocol.constants import RLIP_PROTOCOL_VERSION, ErrorCodes
from .dispatcher import RLIPDispatcher
from .session import SessionManager

log = logging.getLogger(__name__)


def create_app(
    env_registry: EnvironmentRegistry | None = None,
    max_instances: int = 64,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """
    Construct and return the FastAPI application.

    Parameters
    ----------
    env_registry:
        Registry to use.  Defaults to the global singleton.
    max_instances:
        Maximum number of concurrently active environment instances.
    cors_origins:
        Allowed CORS origins.  Defaults to ``["*"]`` (all).
    """
    reg = env_registry or default_registry
    session = SessionManager(max_instances=max_instances)
    dispatcher = RLIPDispatcher(registry=reg, session=session)

    app = FastAPI(
        title="rlbridge Server",
        description=(
            "Reinforcement Learning Interaction Protocol - "
            "JSON-RPC 2.0 over HTTP"
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "server": "rlbridge", "version": __version__}

    @app.get("/info")
    async def info() -> dict[str, Any]:
        return {
            "server_name": "rlbridge Server",
            "server_version": __version__,
            "protocol_version": RLIP_PROTOCOL_VERSION,
            "registered_environments": len(reg),
            "active_instances": len(session),
            "capabilities": {
                "environments": True,
                "rendering": True,
                "sampling": True,
            },
        }

    @app.get("/environments")
    async def list_environments(
        tags: str = "",
        namespace: str = "",
    ) -> dict[str, Any]:
        """Quick REST shorthand to browse available environments."""
        tag_filter = [t.strip() for t in tags.split(",") if t.strip()]
        envs = reg.list_environments(
            tags=tag_filter or None,
            namespace=namespace or None,
        )
        return {"environments": [e.model_dump() for e in envs], "total": len(envs)}

    @app.post("/rpc")
    async def rpc_endpoint(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": ErrorCodes.PARSE_ERROR,
                        "message": "Invalid JSON in request body",
                    },
                },
            )

        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": ErrorCodes.INVALID_REQUEST,
                        "message": "Request body must be a JSON object",
                    },
                },
            )

        response = dispatcher.dispatch(body)
        status = 200 if response.get("error") is None else 200  # JSON-RPC always 200
        return JSONResponse(content=response, status_code=status)

    @app.post("/rpc/batch")
    async def rpc_batch_endpoint(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": ErrorCodes.PARSE_ERROR,
                        "message": "Invalid JSON in request body",
                    },
                },
            )

        if not isinstance(body, list):
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": ErrorCodes.INVALID_REQUEST,
                        "message": "Batch endpoint requires a JSON array",
                    },
                },
            )

        responses = [dispatcher.dispatch(req) for req in body if isinstance(req, dict)]
        return JSONResponse(content=responses)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        count = session.close_all()
        log.info("Closed %d environment instance(s) on shutdown.", count)

    return app
