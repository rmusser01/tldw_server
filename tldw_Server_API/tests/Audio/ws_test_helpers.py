from contextlib import contextmanager
import os

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@contextmanager
def ws_client_without_lifespan(app):
    """Create a TestClient without entering the app lifespan context."""
    previous_query_token_auth = os.environ.get("AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH")
    os.environ["AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH"] = "1"
    try:
        from tldw_Server_API.app.services.app_lifecycle import reset_lifecycle_state

        reset_lifecycle_state(app)
    except (ImportError, AttributeError):
        pass
    try:
        from tldw_Server_API.app.core.Streaming import streams

        streams._STREAM_METRICS_REGISTERED = False
    except (ImportError, AttributeError):
        pass
    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()
        if previous_query_token_auth is None:
            os.environ.pop("AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH", None)
        else:
            os.environ["AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH"] = previous_query_token_auth


@contextmanager
def ws_session_or_skip(ws, *, reason: str = "audio WebSocket endpoint not available in this build"):
    """Enter a TestClient websocket session or skip when it disconnects on entry."""
    try:
        session = ws.__enter__()
    except WebSocketDisconnect:
        pytest.skip(reason)
    try:
        yield session
    finally:
        ws.__exit__(None, None, None)


def scripted_websocket(*frames, headers=None, query: bytes = b"", path: str = "/api/v1/audio/stream/transcribe"):
    """A real Starlette WebSocket whose client connects, sends ``frames`` (text, or dicts as JSON),
    then waits. Returns the socket and the messages the server sent; ``socket.scope["_inbox"]``
    holds what the server left unread."""
    import asyncio
    import json

    from starlette.types import Message
    from starlette.websockets import WebSocket

    inbox: list[Message] = [{"type": "websocket.connect"}]
    inbox += [{"type": "websocket.receive", "text": f if isinstance(f, str) else json.dumps(f)} for f in frames]
    sent: list[Message] = []

    async def receive() -> Message:
        if inbox:
            return inbox.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(message: Message) -> None:
        sent.append(message)

    scope = {
        "type": "websocket",
        "path": path,
        "headers": list(headers or []),
        "query_string": query,
        "client": ("127.0.0.1", 50000),
        "_inbox": inbox,
    }
    return WebSocket(scope, receive=receive, send=send), sent
