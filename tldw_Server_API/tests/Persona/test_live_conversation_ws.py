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


@pytest.fixture(autouse=True)
def fresh_persona_runtime(monkeypatch):
    from tldw_Server_API.app.core.Persona.session_manager import SessionManager

    manager = SessionManager()
    monkeypatch.setattr(ep, "get_session_manager", lambda: manager)
    return manager


@pytest.fixture
def owned_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_persona_runtime) -> Any:
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


@pytest.fixture
def persistent_setup_session(tmp_path, monkeypatch):
    async def auth(*_args, **_kwargs):
        return "1", True, True

    monkeypatch.setattr(ep, "_resolve_authenticated_user_id", auth)
    sid = "persistent-setup-test"
    _seed_persona_session(tmp_path, monkeypatch, user_id="1", session_id=sid, mode="persistent_scoped")
    return sid


@pytest.mark.parametrize("outage", [False, True])
def test_persona_setup_persistent_typed_reply_and_outage_retry(monkeypatch, persistent_setup_session, outage):
    calls = []

    async def answer(**kwargs):
        calls.append(kwargs)
        if outage and len(calls) == 1:
            raise live_conversation.PersonaConversationError("Chat provider unavailable. Retry when ready.")
        return "ORBIT-742"

    monkeypatch.setattr(live_conversation, "complete_persona_conversation", answer)
    with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as ws:
        first_id = "setup-743" if outage else "setup-742"
        ws.send_json(
            {
                "type": "user_message",
                "session_id": persistent_setup_session,
                "client_message_id": first_id,
                "text": "Reply with exactly ORBIT-743" if outage else "Reply with exactly ORBIT-742",
                "use_companion_context": False,
            }
        )
        event = _recv_until(
            ws,
            lambda value: (
                value.get("event") == "assistant_delta" or value.get("reason_code") == "CONVERSATION_UNAVAILABLE"
            ),
        )
        # Pytest behavior checks use assertions; application guards remain enabled.
        assert event["session_id"] == persistent_setup_session  # nosec B101
        assert event["client_message_id"] == first_id  # nosec B101
        if outage:
            assert event["reason_code"] == "CONVERSATION_UNAVAILABLE"  # nosec B101
            assert event["event"] == "notice"  # nosec B101
            assert "Retry" in event["message"]  # nosec B101
            ws.send_json(
                {
                    "type": "user_message",
                    "session_id": persistent_setup_session,
                    "client_message_id": "setup-742-retry",
                    "text": "Reply with exactly ORBIT-742",
                    "use_companion_context": False,
                }
            )
            event = _recv_until(ws, lambda value: value.get("event") == "assistant_delta")
            assert event["session_id"] == persistent_setup_session  # nosec B101
            assert event["client_message_id"] == "setup-742-retry"  # nosec B101
        assert event["event"] == "assistant_delta"  # nosec B101
        assert event["text_delta"] == "ORBIT-742"  # nosec B101
        assert event["done"] is True  # nosec B101
    assert len(calls) == (2 if outage else 1)  # nosec B101


def test_live_turns_survive_manager_restart_without_regeneration(monkeypatch, owned_session):
    """An acknowledged real producer turn must survive replacement of runtime state."""
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.Persona.session_manager import SessionManager

    calls = []
    monkeypatch.setattr(ep, "_get_persona_rbac_flags", lambda: (True, True))

    async def answer(**kwargs):
        calls.append(kwargs)
        return "ORBIT-743"

    async def user():
        return User(id=1, username="restart-owner", email=None, is_active=True)

    monkeypatch.setattr(live_conversation, "complete_persona_conversation", answer)
    db = CharactersRAGDB(str(DatabasePaths.get_chacha_db_path(1)), client_id="persona-durable-restart-test")
    fastapi_app.dependency_overrides[get_request_user] = user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    try:
        with TestClient(fastapi_app) as client:
            with client.websocket_connect("/api/v1/persona/stream") as ws:
                ws.send_json({
                    "type": "user_message", "session_id": owned_session,
                    "client_message_id": "durable-restart-turn",
                    "text": "The supplied code is ORBIT-743. Reply only with the code.",
                    "use_memory_context": False, "use_companion_context": False,
                })
                event = _recv_until(ws, lambda row: row.get("event") == "assistant_delta" and row.get("done") is True)
                assert event["text_delta"] == "ORBIT-743"  # nosec B101 -- Pytest behavior assertion.
            before = client.get("/api/v1/persona/sessions/" + owned_session).json()
            assert before["turn_count"] == 2  # nosec B101 -- Pytest behavior assertion.
            old_ids = [row["turn_id"] for row in before["turns"]]
            assert len(set(old_ids)) == 2  # nosec B101 -- Pytest behavior assertion.
            restarted = SessionManager()
            monkeypatch.setattr(ep, "get_session_manager", lambda: restarted)
            after = client.get("/api/v1/persona/sessions/" + owned_session)
            assert after.status_code == 200  # nosec B101 -- Pytest behavior assertion.
            assert after.json()["turns"] == before["turns"]  # nosec B101 -- Exact persisted identity and content.
            rows = client.get("/api/v1/persona/sessions").json()
            assert next(row for row in rows if row["session_id"] == owned_session)["turn_count"] == 2  # nosec B101
            exported = client.get("/api/v1/persona/sessions/" + owned_session + "/export")
            assert exported.status_code == 200  # nosec B101
            assert [row["turn_id"] for row in exported.json()["turns"]] == old_ids  # nosec B101
            assert [row["content"] for row in exported.json()["turns"]] == [row["content"] for row in before["turns"]]  # nosec B101
            resumed = client.post("/api/v1/persona/session", json={
                "persona_id": "research_assistant", "resume_session_id": owned_session,
            })
            assert resumed.status_code == 200  # nosec B101 -- Pytest behavior assertion.
            assert resumed.json()["session_id"] == owned_session  # nosec B101 -- Pytest behavior assertion.
            assert restarted.list_turns(session_id=owned_session, user_id="1") == before["turns"]  # nosec B101 -- No duplication or regenerated IDs.
            assert len(calls) == 1  # nosec B101 -- Reading/resuming must not call the provider.
    finally:
        fastapi_app.dependency_overrides.clear()
        db.close_all_connections()
        db.close_connection()


def test_failed_assistant_storage_never_emits_completed_reply(monkeypatch, owned_session):
    from tldw_Server_API.app.core.DB_Management.chacha.persona_transcript_store import PersonaTranscriptStore
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError
    from tldw_Server_API.app.core.Persona.session_manager import SessionManager

    manager = SessionManager()
    monkeypatch.setattr(ep, "get_session_manager", lambda: manager)
    original = PersonaTranscriptStore.append_persona_transcript_turn

    def append(self, **kwargs):
        if kwargs["turn"]["role"] == "assistant":
            raise CharactersRAGDBError("Injected transcript storage failure")
        return original(self, **kwargs)

    async def answer(**kwargs):
        return "Must not acknowledge this answer"

    monkeypatch.setattr(PersonaTranscriptStore, "append_persona_transcript_turn", append)
    monkeypatch.setattr(live_conversation, "complete_persona_conversation", answer)
    events = []
    with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as ws:
        ws.send_json({"type": "user_message", "session_id": owned_session, "text": "Hi",
                      "use_memory_context": False, "use_companion_context": False})
        _recv_until(ws, lambda event: events.append(event) or event.get("reason_code") == "USER_TURN_FAILED")
    assert not any(event.get("event") == "assistant_delta" for event in events)  # nosec B101
    turns = manager.list_turns(session_id=owned_session, user_id="1")
    assert [turn["role"] for turn in turns] == ["user"]  # nosec B101 -- No false runtime completion either.


def test_held_provider_cannot_resurrect_expired_question(monkeypatch, owned_session):
    from datetime import datetime, timedelta, timezone

    from tldw_Server_API.app.core.DB_Management.chacha.persona_transcript_store import PersonaTranscriptStore
    from tldw_Server_API.app.core.Persona.session_manager import SessionManager

    manager = SessionManager()
    monkeypatch.setattr(ep, "get_session_manager", lambda: manager)
    now = [datetime.now(timezone.utc)]
    monkeypatch.setattr(PersonaTranscriptStore, "_now", staticmethod(lambda: now[0]))
    started = threading.Event()
    release = threading.Event()

    async def answer(**kwargs):
        started.set()
        assert await asyncio.to_thread(release.wait, 5)  # nosec B101 -- Bounded held-provider fixture.
        return "ORBIT-745"

    monkeypatch.setattr(live_conversation, "complete_persona_conversation", answer)
    events = []
    try:
        with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as ws:
            ws.send_json({"type": "user_message", "session_id": owned_session, "text": "Old question",
                          "client_message_id": "expired", "use_memory_context": False, "use_companion_context": False})
            assert started.wait(5)  # nosec B101
            now[0] += timedelta(days=2)
            release.set()
            _recv_until(ws, lambda event: events.append(event) or event.get("reason_code") == "USER_TURN_FAILED")
            assert not any(event.get("event") == "assistant_delta" for event in events)  # nosec B101
            ws.send_json({"type": "user_message", "session_id": owned_session, "text": "New legitimate question",
                          "client_message_id": "fresh", "use_memory_context": False, "use_companion_context": False})
            event = _recv_until(ws, lambda item: item.get("event") == "assistant_delta" and item.get("done"))
            assert event["client_message_id"] == "fresh"  # nosec B101
            assert event["text_delta"] == "ORBIT-745"  # nosec B101
        turns = manager.list_turns(session_id=owned_session, user_id="1")
        assert [turn["content"] for turn in turns] == ["New legitimate question", "ORBIT-745"]  # nosec B101
    finally:
        release.set()


def test_cancellation_joins_owned_storage_and_does_not_emit_late_completion(monkeypatch, owned_session):
    from tldw_Server_API.app.core.DB_Management.chacha.persona_transcript_store import PersonaTranscriptStore
    from tldw_Server_API.app.core.Persona.session_manager import SessionManager

    manager = SessionManager()
    monkeypatch.setattr(ep, "get_session_manager", lambda: manager)
    original = PersonaTranscriptStore.append_persona_transcript_turn
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def append(self, **kwargs):
        if kwargs["turn"]["role"] == "assistant":
            entered.set()
            assert release.wait(5)  # nosec B101 -- Bounded storage-worker hold.
            try:
                return original(self, **kwargs)
            finally:
                finished.set()
        return original(self, **kwargs)

    async def answer(**kwargs):
        return "Saved before cancellation finished"

    monkeypatch.setattr(PersonaTranscriptStore, "append_persona_transcript_turn", append)
    monkeypatch.setattr(live_conversation, "complete_persona_conversation", answer)
    events = []
    try:
        with TestClient(fastapi_app) as client:
            with client.websocket_connect("/api/v1/persona/stream") as ws:
                ws.send_json({"type": "user_message", "session_id": owned_session, "text": "First",
                              "client_message_id": "cancelled-storage", "use_memory_context": False,
                              "use_companion_context": False})
                assert entered.wait(5)  # nosec B101
                ws.send_json({"type": "cancel", "session_id": owned_session})
                _recv_until(ws, lambda event: events.append(event) or event.get("reason_code") == "PLAN_CANCELLED")
                assert not finished.is_set()  # nosec B101 -- Cancellation is not permission to abandon its worker.
                release.set()
            assert finished.is_set()  # nosec B101 -- WebSocket teardown joined the exact storage operation.
        assert not any(event.get("event") == "assistant_delta" for event in events)  # nosec B101
        assert [turn["role"] for turn in manager.list_turns(session_id=owned_session, user_id="1")] == ["user"]  # nosec B101
    finally:
        release.set()
