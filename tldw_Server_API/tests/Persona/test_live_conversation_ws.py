"""Owned persistent Buddy turns use conversation with cancellable publication."""

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import persona as ep
from tldw_Server_API.app.core.Persona import live_conversation
from tldw_Server_API.tests.Persona.test_persona_ws import (
    _recv_until,
    _seed_persona_session,
    fastapi_app,
)


@pytest.fixture
def owned_session(tmp_path, monkeypatch):
    async def auth(*args, **kwargs):
        return "1", True, True

    monkeypatch.setattr(ep, "_resolve_authenticated_user_id", auth)
    sid = "live-conversation-test"
    _seed_persona_session(tmp_path, monkeypatch, user_id="1", session_id=sid, mode="session_scoped")
    return sid


def test_owned_greeting_returns_correlated_reply_without_plan(monkeypatch, owned_session):
    calls = []

    async def answer(**kwargs):
        calls.append(kwargs)
        return "Hello from Migu"

    monkeypatch.setattr(live_conversation, "complete_persona_conversation", answer)
    with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as ws:
        ws.send_json(
            {
                "type": "user_message",
                "session_id": owned_session,
                "client_message_id": "greeting",
                "text": "Hi Migu",
                "use_companion_context": False,
            }
        )
        event = _recv_until(ws, lambda event: event.get("event") in {"assistant_delta", "tool_plan"})
        assert event["event"] == "assistant_delta"
        assert event["client_message_id"] == "greeting"
        assert event["text_delta"] == "Hello from Migu"
    assert calls[0]["system_prompt"].startswith("Helper")


def test_cancel_then_retry_drops_late_provider_output(monkeypatch, owned_session):
    started = threading.Event()
    release_old = asyncio.Event()
    count = 0

    async def answer(**kwargs):
        nonlocal count
        count += 1
        if count == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release_old.wait()
                return "STALE ANSWER"
        release_old.set()
        await asyncio.sleep(0)
        return "Fresh answer"

    monkeypatch.setattr(live_conversation, "complete_persona_conversation", answer)
    with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as ws:
        ws.send_json(
            {
                "type": "user_message",
                "session_id": owned_session,
                "client_message_id": "old",
                "text": "Hi",
                "use_companion_context": False,
            }
        )
        assert started.wait(5)
        ws.send_json({"type": "cancel", "session_id": owned_session})
        _recv_until(ws, lambda event: event.get("reason_code") == "PLAN_CANCELLED")
        ws.send_json(
            {
                "type": "user_message",
                "session_id": owned_session,
                "client_message_id": "new",
                "text": "Hello again",
                "use_companion_context": False,
            }
        )
        event = _recv_until(ws, lambda event: event.get("event") == "assistant_delta")
        assert event["client_message_id"] == "new"
        assert event["text_delta"] == "Fresh answer"


def test_explicit_search_retains_plan(monkeypatch, owned_session):
    async def forbidden(**kwargs):
        pytest.fail("Tool request bypassed plan review")

    monkeypatch.setattr(live_conversation, "complete_persona_conversation", forbidden)
    with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as ws:
        ws.send_json(
            {
                "type": "user_message",
                "session_id": owned_session,
                "client_message_id": "search",
                "text": "Search my notes for cats",
                "use_companion_context": False,
            }
        )
        event = _recv_until(ws, lambda event: event.get("event") == "tool_plan")
        assert event["client_message_id"] == "search"
        assert event["steps"][0]["tool"] == "rag_search"
