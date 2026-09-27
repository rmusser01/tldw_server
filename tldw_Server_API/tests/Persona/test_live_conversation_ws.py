"""Owned persistent Buddy turns use conversation with cancellable publication."""

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import persona as ep
from tldw_Server_API.app.core.Persona import live_conversation
from tldw_Server_API.tests.Persona.test_persona_ws import (
    _recv_until,
    _seed_persona_session,
    fastapi_app,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def owned_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    async def auth(*args: Any, **kwargs: Any) -> Any:
        return "1", True, True

    monkeypatch.setattr(ep, "_resolve_authenticated_user_id", auth)
    sid = "live-conversation-test"
    _seed_persona_session(tmp_path, monkeypatch, user_id="1", session_id=sid, mode="session_scoped")
    return sid


def test_owned_greeting_returns_correlated_reply_without_plan(
    monkeypatch: pytest.MonkeyPatch, owned_session: str
) -> None:
    calls = []

    async def answer(**kwargs: Any) -> Any:
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


def test_cancel_then_retry_drops_late_provider_output(monkeypatch: pytest.MonkeyPatch, owned_session: str) -> None:
    started = threading.Event()
    release_old = asyncio.Event()
    count = 0

    async def answer(**kwargs: Any) -> Any:
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


def test_explicit_search_retains_plan(monkeypatch: pytest.MonkeyPatch, owned_session: str) -> None:
    async def forbidden(**kwargs: Any) -> Any:
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


def test_unexpected_turn_failure_logs_safe_correlated_diagnostic(
    monkeypatch: pytest.MonkeyPatch, owned_session: str
) -> None:
    from loguru import logger

    records = []
    sink = logger.add(lambda message: records.append(message.record))

    async def fail(**kwargs: Any) -> Any:
        raise RuntimeError("secret-provider-payload")

    monkeypatch.setattr(live_conversation, "complete_persona_conversation", fail)
    try:
        with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as ws:
            ws.send_json(
                {
                    "type": "user_message",
                    "session_id": owned_session,
                    "client_message_id": "failed-turn",
                    "text": "Hi",
                    "use_companion_context": False,
                }
            )
            event = _recv_until(ws, lambda value: value.get("reason_code") == "USER_TURN_FAILED")
            assert "secret-provider-payload" not in event["message"]
        failures = [record for record in records if record["extra"].get("reason_code") == "USER_TURN_FAILED"]
        assert len(failures) == 1
        assert failures[0]["extra"]["client_message_id"] == "failed-turn"
        assert failures[0]["extra"]["error_type"] == "RuntimeError"
        assert "secret-provider-payload" not in str(failures[0])
    finally:
        logger.remove(sink)
