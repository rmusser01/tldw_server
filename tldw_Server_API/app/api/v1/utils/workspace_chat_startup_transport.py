"""Strict-only startup body bounds and content-free validation responses."""

from __future__ import annotations

import email.message
import re
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import ValidationError
from starlette.responses import JSONResponse, Response
from starlette.types import Message

from tldw_Server_API.app.api.v1.schemas.workspace_chat_startup_schemas import (
    STARTUP_BODY_BYTES_MAX,
    STARTUP_IDEMPOTENCY_KEY_PATTERN,
    STARTUP_TEXT_BYTE_LIMITS,
    WorkspaceChatStartupRequest,
)


def strict_startup_validation_detail(error: RequestValidationError) -> list[dict[str, Any]]:
    """Return fixed messages and bounded static locations, never raw input."""
    allowed = set(STARTUP_TEXT_BYTE_LIMITS) | {
        "body", "header", "query", "scope_type",
        "workspace_assistant_selection", "workspace_assistant_default_version",
        "idempotency-key", "Idempotency-Key",
    }
    details = []
    for item in error.errors()[:32]:
        location = [part for part in item.get("loc", ()) if isinstance(part, str) and part in allowed][:4]
        details.append({
            "loc": location or ["body"],
            "msg": "Invalid Workspace startup request",
            "type": "value_error",
        })
    return details


class WorkspaceStartupRoute(APIRoute):
    """Bound raw input before FastAPI parsing without changing legacy routes."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        """Replay the bounded body once, then preserve the original receive stream."""
        handler = super().get_route_handler()

        async def bounded_handler(request: Request) -> Response:
            """Reject oversized or invalid UTF-8 input before dependencies and parsing."""
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > STARTUP_BODY_BYTES_MAX:
                    return JSONResponse(status_code=413, content={
                        "detail": {"code": "workspace_chat_startup_body_too_large"},
                    })
                body.extend(chunk)
            raw = bytes(body)
            receive = request.receive
            delivered = False

            async def replay_receive() -> Message:
                """Deliver buffered input once without hiding subsequent disconnects."""
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": raw, "more_body": False}
                return await receive()

            try:
                if request.scope.get("query_string"):
                    raise RequestValidationError([{"loc": ("query",)}])
                keys = request.headers.getlist("idempotency-key")
                if len(keys) != 1 or re.fullmatch(STARTUP_IDEMPOTENCY_KEY_PATTERN, keys[0]) is None:
                    raise RequestValidationError([{"loc": ("header", "idempotency-key")}])
                media_type = email.message.Message()
                media_type["content-type"] = request.headers.get("content-type", "")
                subtype = media_type.get_content_subtype()
                if media_type.get_content_maintype() != "application" or not (
                    subtype == "json" or subtype.endswith("+json")
                ):
                    raise RequestValidationError([{"loc": ("body",)}])
                try:
                    # FastAPI initializes DB dependencies before reporting model
                    # errors; validate the same model first, on UTF-8 text only.
                    WorkspaceChatStartupRequest.model_validate_json(raw.decode("utf-8", errors="strict"))
                except UnicodeDecodeError:
                    raise RequestValidationError([{"loc": ("body",)}]) from None
                except ValidationError as error:
                    raise RequestValidationError([
                        {"loc": ("body", *item.get("loc", ()))} for item in error.errors()[:32]
                    ]) from None
                return await handler(Request(request.scope, receive=replay_receive))
            except RequestValidationError as error:
                return JSONResponse(status_code=422, content={"detail": strict_startup_validation_detail(error)})

        return bounded_handler
