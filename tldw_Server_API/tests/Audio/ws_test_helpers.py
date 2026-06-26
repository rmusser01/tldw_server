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
