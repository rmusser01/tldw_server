"""Transport bounds for a chat import: body size and validation responses (D7 P8).

An import carries a whole chat, images included, so its body is far larger than
other chat requests. Two defaults of the framework do not fit that:

* FastAPI buffers a JSON body without any limit. This route stops reading at
  ``CHAT_IMPORT_MAX_BODY_BYTES`` (64 MiB by default) and answers 413 without
  parsing anything.
* A validation error echoes the rejected input, which for a missing top-level
  field is the entire body. This route reports where and why the body is
  invalid and leaves the content out. The response is ASCII-escaped, because
  the rejected text may hold half of a surrogate pair that cannot be encoded.

The body is still read and parsed before the request is authenticated, as for
every JSON route here: the framework reads a declared body before it resolves
dependencies. The limit bounds that work; it does not move it behind the login.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse, Response
from starlette.types import Message

from tldw_Server_API.app.core.config import settings

DEFAULT_CHAT_IMPORT_MAX_BODY_BYTES = 64 * 1024 * 1024
CHAT_IMPORT_BODY_TOO_LARGE = "import_body_too_large"
CHAT_IMPORT_MAX_VALIDATION_ERRORS = 100
_MAX_LOCATION_PARTS = 8
_MAX_TEXT_CHARS = 300


def chat_import_max_body_bytes() -> int:
    """Return the raw-body limit for one import, in bytes.

    ``CHAT_IMPORT_MAX_BODY_BYTES`` is read from the environment first and then
    from settings; a missing, non-numeric or non-positive value means the default.
    """
    for raw in (os.getenv("CHAT_IMPORT_MAX_BODY_BYTES"), settings.get("CHAT_IMPORT_MAX_BODY_BYTES")):
        try:
            configured = int(raw)
        except (TypeError, ValueError):
            continue
        if configured > 0:
            return configured
    return DEFAULT_CHAT_IMPORT_MAX_BODY_BYTES


def _too_large(limit: int) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={
            "detail": {
                "error_code": CHAT_IMPORT_BODY_TOO_LARGE,
                "message": f"This chat is larger than the {limit} bytes one import may carry.",
                "max_bytes": limit,
            }
        },
    )


def chat_import_validation_detail(error: RequestValidationError) -> list[dict[str, Any]]:
    """Return ``type``, ``loc`` and ``msg`` for each validation error, without the rejected input."""
    details: list[dict[str, Any]] = []
    for item in error.errors()[:CHAT_IMPORT_MAX_VALIDATION_ERRORS]:
        location = [
            part if isinstance(part, int) else str(part)[:_MAX_TEXT_CHARS]
            for part in tuple(item.get("loc") or ())[:_MAX_LOCATION_PARTS]
        ]
        details.append(
            {
                "type": str(item.get("type") or "value_error"),
                "loc": location or ["body"],
                "msg": str(item.get("msg") or "Invalid value")[:_MAX_TEXT_CHARS],
            }
        )
    return details


def _invalid(error: RequestValidationError) -> Response:
    body = json.dumps({"detail": chat_import_validation_detail(error)}, ensure_ascii=True, separators=(",", ":"))
    return Response(content=body, status_code=422, media_type="application/json")


class ChatImportRoute(APIRoute):
    """Read at most the configured number of body bytes, then hand the body to FastAPI."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        """Wrap the normal handler; a valid request within the limit is handled unchanged."""
        handler = super().get_route_handler()

        async def bounded_handler(request: Request) -> Response:
            """Refuse an oversized body from its declared length or while streaming it."""
            limit = chat_import_max_body_bytes()
            declared = request.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > limit:
                return _too_large(limit)
            chunks: list[bytes] = []
            received = 0
            async for chunk in request.stream():
                received += len(chunk)
                if received > limit:
                    return _too_large(limit)
                chunks.append(chunk)
            raw = b"".join(chunks)
            # Keep one copy of a large body for the rest of the request, not two.
            del chunks
            receive = request.receive
            delivered = False

            async def replay_receive() -> Message:
                """Deliver the buffered body once without hiding a later disconnect."""
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": raw, "more_body": False}
                return await receive()

            try:
                return await handler(Request(request.scope, receive=replay_receive))
            except RequestValidationError as error:
                return _invalid(error)

        return bounded_handler
