"""
Global exception handlers for the FastAPI application.

Extracted from main.py to enable unit testing and establish the
canonical error response format for Phase 3.1 migration.

Response format (target for all error responses)::

    {
        "error": {
            "code": "internal_server_error",
            "message": "Human-readable description",
            "request_id": "<canonical request id or generated UUID>"
        }
    }
"""

from __future__ import annotations

import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.requests import ClientDisconnect


def _get_request_id(request: Request) -> str:
    """Extract or generate a request ID for error tracing."""
    state = getattr(request, "state", None)
    request_id = getattr(state, "request_id", None)
    if request_id:
        return request_id

    header_request_id = request.headers.get("X-Request-ID")
    if header_request_id:
        return header_request_id

    generated_request_id = str(uuid.uuid4())
    if state is not None:
        try:
            state.request_id = generated_request_id
        except (AttributeError, TypeError):
            logger.debug("Unable to persist generated request_id onto request.state")
    return generated_request_id


async def global_unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Catch-all handler for unhandled exceptions.

    Returns a structured JSON error instead of Starlette's default HTML 500.
    """
    if isinstance(exc, ClientDisconnect):
        return await client_disconnect_handler(request, exc)

    request_id = _get_request_id(request)
    logger.opt(exception=exc).error(
        "Unhandled exception on {method} {path} (request_id={rid}): {exc}",
        method=request.method,
        path=request.url.path,
        rid=request_id,
        exc=exc,
    )
    return JSONResponse(
        status_code=500,
        headers={"X-Request-ID": request_id},
        content={
            "error": {
                "code": "internal_server_error",
                "message": "Internal server error",
                "request_id": request_id,
            },
        },
    )


async def client_disconnect_handler(
    request: Request,
    exc: ClientDisconnect,
) -> JSONResponse:
    """Handler for client disconnect exceptions."""
    request_id = _get_request_id(request)
    logger.debug(
        "Client disconnected during {method} {path} (request_id={rid})",
        method=request.method,
        path=request.url.path,
        rid=request_id,
    )
    return JSONResponse(
        status_code=499,
        headers={"X-Request-ID": request_id},
        content={
            "error": {
                "code": "client_disconnected",
                "message": "Client disconnected",
                "request_id": request_id,
            },
        },
    )


__all__ = [
    "client_disconnect_handler",
    "global_unhandled_exception_handler",
]
