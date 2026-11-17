"""Observe bounded generation disconnects outside BaseHTTPMiddleware receive polling."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tldw_Server_API.app.core.Chat.bounded_daemon import await_owned_worker

_SCOPE_KEY = "tldw.slides_generation_disconnect"
_PATHS = frozenset(
    "/api/v1/slides/generate" + suffix for suffix in ("", "/from-chat", "/from-media", "/from-notes", "/from-rag")
) | frozenset({"/api/v1/moderation/classify"})


async def generation_disconnected(request: Request) -> bool:
    """Read the raw observer receipt; retain compatibility with standalone routers."""
    state = request.scope.get(_SCOPE_KEY)
    if state is None:
        return await request.is_disconnected()
    await state["started"].wait()
    if state["error"] is not None:
        raise state["error"]
    return state["closed"].is_set()


class SlidesGenerationDisconnectMiddleware:
    """Own one disconnect receive after body consumption on the fixed generation POST routes."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") not in _PATHS:
            await self.app(scope, receive, send)
            return
        state: dict[str, Any] = {"closed": asyncio.Event(), "started": asyncio.Event(), "error": None}
        scope[_SCOPE_KEY] = state
        reader: asyncio.Task[None] | None = None
        body_complete = False

        async def watch_disconnect() -> None:
            state["started"].set()
            try:
                message = await receive()
                if message["type"] != "http.disconnect":
                    raise RuntimeError("Unexpected ASGI message after completed generation request body")
                state["closed"].set()
            except Exception as error:
                state["error"] = error
                state["closed"].set()
                raise

        async def observed_receive() -> Message:
            nonlocal body_complete, reader
            if body_complete:
                await state["closed"].wait()
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.disconnect":
                state["closed"].set()
                state["started"].set()
            elif message["type"] == "http.request" and not message.get("more_body", False):
                body_complete = True
                reader = asyncio.create_task(watch_disconnect())
            return message

        try:
            await self.app(scope, observed_receive, send)
        finally:
            primary = sys.exception()
            if reader is not None:
                reader.cancel()
                outcome = (await await_owned_worker(asyncio.gather(reader, return_exceptions=True)))[0]
                if isinstance(outcome, Exception):
                    if primary is None:
                        raise outcome
                    primary.add_note("Slides disconnect observer also failed: " + type(outcome).__name__)
