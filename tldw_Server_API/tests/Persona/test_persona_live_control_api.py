"""Tests for Persona Buddy live-control REST and WebSocket behavior."""

import json
import queue
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Persona import live_control as live_control_module
from tldw_Server_API.app.core.Persona.exemplar_prompt_assembly import PersonaExemplarPromptAssembly
from tldw_Server_API.app.core.Persona.exemplar_runtime import PersonaExemplarRuntimeContext
from tldw_Server_API.app.core.Persona.live_control import persona_live_stream_registry
from tldw_Server_API.app.core.Persona.session_manager import SessionManager

pytestmark = pytest.mark.unit

fastapi_app = FastAPI()
fastapi_app.include_router(persona_ep.router, prefix="/api/v1/persona")


@pytest.fixture()
def persona_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "persona_live_control.db"), client_id="persona-live-control-tests")
    yield db
    db.close_connection()


@pytest.fixture(autouse=True)
def _clear_overrides_and_registry():
    fastapi_app.dependency_overrides.clear()
    persona_live_stream_registry.clear()
    yield
    fastapi_app.dependency_overrides.clear()
    persona_live_stream_registry.clear()


def _client_for_user(user_id: int, db: CharactersRAGDB):
    async def override_user():
        return User(id=user_id, username=f"user-{user_id}", email=None, is_active=True)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    return TestClient(fastapi_app)


def _create_profile(db: CharactersRAGDB, *, user_id: str, persona_id: str, name: str | None = None) -> str:
    return db.create_persona_profile(
        {
            "id": persona_id,
            "user_id": user_id,
            "name": name or persona_id.replace("_", " ").title(),
            "mode": "session_scoped",
            "system_prompt": f"{persona_id} helper",
            "is_active": True,
        }
    )


def _create_session(
    db: CharactersRAGDB,
    *,
    user_id: str,
    persona_id: str,
    session_id: str,
    status: str = "active",
    surface: str = "api.persona",
    preferences: dict | None = None,
) -> str:
    return db.create_persona_session(
        {
            "id": session_id,
            "persona_id": persona_id,
            "user_id": user_id,
            "mode": "session_scoped",
            "status": status,
            "scope_snapshot_json": {
                "scope_snapshot_id": f"scope-{session_id}",
                "audit": {"scope_snapshot_id": f"scope-{session_id}"},
            },
            "preferences_json": preferences or {"use_memory_context": True},
            "activity_surface": surface,
        }
    )


def _session_ids(payload: dict) -> list[str]:
    return [item["session_id"] for item in payload["sessions"]]


def _recv_until(client, predicate, timeout=2.0):
    start = time.time()
    while time.time() - start < timeout:
        inbox: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

        def _reader() -> None:
            try:
                inbox.put(("ok", client.receive_text()))
            except Exception as exc:  # pragma: no cover - defensive test harness path
                inbox.put(("err", exc))

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        remaining = max(0.01, min(0.1, timeout - (time.time() - start)))
        try:
            status, payload = inbox.get(timeout=remaining)
        except queue.Empty:
            continue
        if status == "err":
            raise payload  # type: ignore[misc]
        try:
            data = json.loads(str(payload))
        except Exception:
            continue
        if predicate(data):
            return data
    raise AssertionError("Expected event not received in time")


def _install_persona_stream_test_stubs(
    monkeypatch,
    manager: SessionManager,
    *,
    user_id: str = "1",
    persisted_turns: list[dict[str, object]] | None = None,
) -> None:
    async def _fake_resolve_authenticated_user_id(*args, **kwargs):
        return user_id, True, True

    def _fake_persist_persona_turn(**kwargs: object) -> bool:
        if persisted_turns is not None:
            persisted_turns.append(kwargs)
        return True

    async def _fake_resolve_persona_exemplar_runtime_context(**kwargs: object) -> PersonaExemplarRuntimeContext:
        return PersonaExemplarRuntimeContext(
            assembly=PersonaExemplarPromptAssembly(
                sections=[],
                selected_exemplars=[],
                rejected_exemplars=[],
            ),
            selection_metadata={},
        )

    monkeypatch.setattr(persona_ep, "_resolve_authenticated_user_id", _fake_resolve_authenticated_user_id)
    monkeypatch.setattr(persona_ep, "_open_persona_ws_db", lambda _user_id: None)
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)
    monkeypatch.setattr(persona_ep, "persist_persona_turn", _fake_persist_persona_turn)
    monkeypatch.setattr(persona_ep, "retrieve_top_memories", lambda **kwargs: [])
    monkeypatch.setattr(persona_ep, "load_companion_context", lambda **kwargs: {})
    monkeypatch.setattr(
        persona_ep,
        "resolve_persona_exemplar_runtime_context",
        _fake_resolve_persona_exemplar_runtime_context,
    )


def _session_summary(payload: dict, session_id: str) -> dict:
    return next(item for item in payload["sessions"] if item["session_id"] == session_id)


def test_live_sessions_requires_auth():
    with TestClient(fastapi_app) as client:
        resp = client.get("/api/v1/persona/live/sessions")
        assert resp.status_code == 401


def test_live_sessions_list_returns_owned_session_summaries(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a", name="Persona A")
    _create_profile(persona_db, user_id="2", persona_id="persona_b", name="Persona B")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-owned")
    _create_session(persona_db, user_id="2", persona_id="persona_b", session_id="sess-other")

    with _client_for_user(1, persona_db) as client:
        resp = client.get("/api/v1/persona/live/sessions")

    assert resp.status_code == 200
    payload = resp.json()
    assert _session_ids(payload) == ["sess-owned"]
    summary = payload["sessions"][0]
    assert summary["persona_id"] == "persona_a"
    assert summary["persona_name"] == "Persona A"
    assert summary["lifecycle"] == "idle"
    assert summary["allowed_actions"] == ["focus", "stop", "send_text_ws"]
    assert summary["capabilities"] == {
        "text": True,
        "voice": False,
        "browser_microphone_required": False,
    }


def test_live_session_create_resume_compatible_reuses_active_session(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())

    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/session",
            json={"persona_id": "research_assistant", "surface": "companion.conversation"},
        )
        assert created.status_code == 200
        session_id = created.json()["session_id"]

        live = client.post(
            "/api/v1/persona/live/sessions",
            json={"persona_id": "research_assistant", "surface": "companion.conversation"},
        )

    assert live.status_code == 200
    assert live.json()["session"]["session_id"] == session_id
    assert live.json()["session"]["is_focused"] is True


def test_live_session_create_resume_compatible_omitted_surface_uses_default_surface(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())

    with _client_for_user(1, persona_db) as client:
        companion = client.post(
            "/api/v1/persona/session",
            json={"persona_id": "research_assistant", "surface": "companion.conversation"},
        )
        assert companion.status_code == 200
        companion_session_id = companion.json()["session_id"]

        live = client.post(
            "/api/v1/persona/live/sessions",
            json={"persona_id": "research_assistant"},
        )

    assert live.status_code == 200
    live_session_id = live.json()["session"]["session_id"]
    assert live_session_id != companion_session_id
    live_row = persona_db.get_persona_session(live_session_id, user_id="1", include_deleted=False)
    assert live_row is not None
    assert live_row["activity_surface"] == "api.persona"


def test_live_session_create_new_honors_idempotency_key(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())

    with _client_for_user(1, persona_db) as client:
        first = client.post(
            "/api/v1/persona/live/sessions",
            json={
                "persona_id": "research_assistant",
                "reuse_policy": "create_new",
                "idempotency_key": " create-key-1 ",
                "surface": "companion.conversation",
            },
        )
        second = client.post(
            "/api/v1/persona/live/sessions",
            json={
                "persona_id": "research_assistant",
                "reuse_policy": "create_new",
                "idempotency_key": "create-key-1",
                "surface": "companion.conversation",
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["session"]["session_id"] == second.json()["session"]["session_id"]


def test_live_session_create_new_unknown_persona_returns_not_found(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())

    with _client_for_user(1, persona_db) as client:
        resp = client.post(
            "/api/v1/persona/live/sessions",
            json={
                "persona_id": "missing_persona",
                "reuse_policy": "create_new",
                "idempotency_key": "unknown-create-key",
                "surface": "companion.conversation",
            },
        )

    assert resp.status_code == 404


def test_live_session_resume_compatible_unknown_persona_returns_not_found(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())

    with _client_for_user(1, persona_db) as client:
        resp = client.post(
            "/api/v1/persona/live/sessions",
            json={
                "persona_id": "missing_persona",
                "reuse_policy": "resume_compatible",
                "surface": "companion.conversation",
            },
        )

    assert resp.status_code == 404


def test_live_session_create_uses_existing_session_materialization(monkeypatch, persona_db: CharactersRAGDB):
    manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)

    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/live/sessions",
            json={
                "persona_id": "research_assistant",
                "reuse_policy": "create_new",
                "surface": "companion.conversation",
            },
        )

    assert created.status_code == 200
    session_id = created.json()["session"]["session_id"]
    row = persona_db.get_persona_session(session_id, user_id="1", include_deleted=False)
    assert row is not None
    assert row["scope_snapshot"]["scope_snapshot_id"]
    assert row["scope_snapshot"]["audit"]["scope_snapshot_id"] == row["scope_snapshot"]["scope_snapshot_id"]
    assert row["preferences"]["use_memory_context"] is True
    assert row["preferences"]["use_companion_context"] is True
    assert "companion_activity_surface" not in row["preferences"]
    assert row["activity_surface"] == "companion.conversation"
    assert (
        manager.get_preferences(session_id=session_id, user_id="1")["companion_activity_surface"]
        == "companion.conversation"
    )


def test_live_session_created_by_live_control_can_resume_existing_session_endpoint(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())

    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/live/sessions",
            json={"persona_id": "research_assistant", "reuse_policy": "create_new"},
        )
        assert created.status_code == 200
        session_id = created.json()["session"]["session_id"]

        resumed = client.post(
            "/api/v1/persona/session",
            json={"persona_id": "research_assistant", "resume_session_id": session_id},
        )

    assert resumed.status_code == 200
    assert resumed.json()["session_id"] == session_id
    assert resumed.json()["scope_snapshot_id"]


def test_live_session_focus_last_write_wins(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-a")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-b")

    with _client_for_user(1, persona_db) as client:
        first = client.post("/api/v1/persona/live/sessions/sess-a/focus")
        second = client.post("/api/v1/persona/live/sessions/sess-b/focus")
        listed = client.get("/api/v1/persona/live/sessions")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["session"]["focus_generation"] > first.json()["session"]["focus_generation"]
    assert listed.json()["focused_session_id"] == "sess-b"


def test_live_session_focus_a_then_b_only_marks_b_focused(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-a")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-b")

    with _client_for_user(1, persona_db) as client:
        assert client.post("/api/v1/persona/live/sessions/sess-a/focus").status_code == 200
        assert client.post("/api/v1/persona/live/sessions/sess-b/focus").status_code == 200
        listed = client.get("/api/v1/persona/live/sessions")

    payload = listed.json()
    focused = [item for item in payload["sessions"] if item["is_focused"]]
    assert [item["session_id"] for item in focused] == ["sess-b"]


def test_live_session_focus_uses_utc_iso_focus_timestamp(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-a")

    with _client_for_user(1, persona_db) as client:
        response = client.post("/api/v1/persona/live/sessions/sess-a/focus")

    assert response.status_code == 200
    focused_at = response.json()["session"]["focused_at"]
    assert focused_at.endswith("+00:00")


def test_live_session_focus_and_stop_return_not_found_for_other_user_session(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-owned")

    with _client_for_user(2, persona_db) as client:
        focus_resp = client.post("/api/v1/persona/live/sessions/sess-owned/focus")
        stop_resp = client.post("/api/v1/persona/live/sessions/sess-owned/stop")

    assert focus_resp.status_code == 404
    assert stop_resp.status_code == 404


def test_live_session_stop_marks_closed_and_clears_focus(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-a")

    with _client_for_user(1, persona_db) as client:
        assert client.post("/api/v1/persona/live/sessions/sess-a/focus").status_code == 200
        stopped = client.post("/api/v1/persona/live/sessions/sess-a/stop")
        listed = client.get("/api/v1/persona/live/sessions")

    assert stopped.status_code == 200
    assert stopped.json()["session"]["status"] == "closed"
    assert stopped.json()["session"]["lifecycle"] == "stopped"
    assert stopped.json()["session"]["is_focused"] is False
    assert listed.json()["focused_session_id"] is None


def test_live_session_stop_terminal_session_is_idempotent(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    _create_session(
        persona_db,
        user_id="1",
        persona_id="persona_a",
        session_id="sess-archived",
        status="archived",
    )

    with _client_for_user(1, persona_db) as client:
        stopped = client.post("/api/v1/persona/live/sessions/sess-archived/stop")

    assert stopped.status_code == 200
    assert stopped.json()["session"]["status"] == "archived"
    row = persona_db.get_persona_session("sess-archived", user_id="1", include_deleted=False)
    assert row is not None
    assert row["status"] == "archived"


def test_live_session_focus_and_stop_preserve_unrelated_preferences(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    _create_session(
        persona_db,
        user_id="1",
        persona_id="persona_a",
        session_id="sess-pref",
        preferences={
            "use_memory_context": False,
            "custom_nested": {"retained": True},
        },
    )

    with _client_for_user(1, persona_db) as client:
        assert client.post("/api/v1/persona/live/sessions/sess-pref/focus").status_code == 200
        focused_row = persona_db.get_persona_session("sess-pref", user_id="1", include_deleted=False)
        assert focused_row is not None
        assert focused_row["preferences"]["use_memory_context"] is False
        assert focused_row["preferences"]["custom_nested"] == {"retained": True}

        assert client.post("/api/v1/persona/live/sessions/sess-pref/stop").status_code == 200
        stopped_row = persona_db.get_persona_session("sess-pref", user_id="1", include_deleted=False)
        assert stopped_row is not None
        assert stopped_row["preferences"]["use_memory_context"] is False
        assert stopped_row["preferences"]["custom_nested"] == {"retained": True}


def test_live_sessions_ignore_stale_focused_closed_session(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    _create_session(
        persona_db,
        user_id="1",
        persona_id="persona_a",
        session_id="sess-closed",
        status="closed",
        preferences={
            "persona_live_control": {
                "focus": {
                    "focused": True,
                    "focused_at": "2026-05-20T00:00:00+00:00",
                    "focus_generation": 999,
                }
            }
        },
    )

    with _client_for_user(1, persona_db) as client:
        listed = client.get("/api/v1/persona/live/sessions")

    payload = listed.json()
    assert payload["focused_session_id"] is None
    assert payload["sessions"][0]["is_focused"] is False


def test_live_session_rest_created_without_stream_is_idle(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())

    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/live/sessions",
            json={"persona_id": "research_assistant", "reuse_policy": "create_new"},
        )

    assert created.status_code == 200
    assert created.json()["session"]["lifecycle"] == "idle"


def test_voice_commit_message_preserves_bounded_client_message_id():
    long_client_message_id = f" {'x' * 140} "

    payload = persona_ep._persona_live_voice_commit_message(
        session_id="sess-voice",
        transcript="hello",
        source="persona_live_voice",
        commit_source="manual",
        client_message_id=long_client_message_id,
    )

    assert payload["session_id"] == "sess-voice"
    assert payload["client_message_id"] == "x" * 128


def test_persona_stream_voice_commit_records_bounded_client_message_id(monkeypatch):
    manager = SessionManager()
    persisted_turns: list[dict[str, object]] = []
    _install_persona_stream_test_stubs(monkeypatch, manager, persisted_turns=persisted_turns)
    session_id = "sess-voice-commit-ws"

    with TestClient(fastapi_app) as client:
        with client.websocket_connect("/api/v1/persona/stream") as ws:
            _ = json.loads(ws.receive_text())
            ws.send_text(
                json.dumps(
                    {
                        "type": "voice_commit",
                        "session_id": session_id,
                        "transcript": "Please summarize this session",
                        "client_message_id": f" {'v' * 140} ",
                    }
                )
            )
            committed = _recv_until(
                ws,
                lambda data: data.get("event") == "notice"
                and data.get("reason_code") == "VOICE_TURN_COMMITTED"
                and data.get("session_id") == session_id,
            )
            plan = _recv_until(
                ws,
                lambda data: data.get("event") == "tool_plan" and data.get("session_id") == session_id,
            )

    assert committed["transcript"] == "Please summarize this session"
    assert plan["session_id"] == session_id
    turns = manager.list_turns(session_id=session_id, user_id="1", limit=10)
    voice_turn = next(turn for turn in turns if turn["type"] == "voice_commit")
    assert voice_turn["metadata"]["client_message_id"] == "v" * 128
    persisted_voice_turn = next(turn for turn in persisted_turns if turn["turn_type"] == "voice_commit")
    assert persisted_voice_turn["metadata"]["client_message_id"] == "v" * 128


def test_live_session_active_stream_presence_is_connected(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-a")
    persona_live_stream_registry.mark_connected(user_id="1", session_id="sess-a")

    with _client_for_user(1, persona_db) as client:
        listed = client.get("/api/v1/persona/live/sessions")

    assert listed.status_code == 200
    assert listed.json()["sessions"][0]["lifecycle"] == "connected"


def test_live_session_idempotency_scans_beyond_first_page(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())

    with _client_for_user(1, persona_db) as client:
        first = client.post(
            "/api/v1/persona/live/sessions",
            json={
                "persona_id": "research_assistant",
                "reuse_policy": "create_new",
                "idempotency_key": "paged-key",
                "surface": "companion.conversation",
            },
        )
        assert first.status_code == 200
        first_session_id = first.json()["session"]["session_id"]
        for idx in range(205):
            filler = client.post(
                "/api/v1/persona/live/sessions",
                json={
                    "persona_id": "research_assistant",
                    "reuse_policy": "create_new",
                    "idempotency_key": f"filler-{idx}",
                    "surface": "companion.conversation",
                },
            )
            assert filler.status_code == 200

        second = client.post(
            "/api/v1/persona/live/sessions",
            json={
                "persona_id": "research_assistant",
                "reuse_policy": "create_new",
                "idempotency_key": "paged-key",
                "surface": "companion.conversation",
            },
        )

    assert second.status_code == 200
    assert second.json()["session"]["session_id"] == first_session_id


def test_live_session_focus_only_touches_target_and_previously_focused(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-a")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-b")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-untouched")

    with _client_for_user(1, persona_db) as client:
        assert client.post("/api/v1/persona/live/sessions/sess-a/focus").status_code == 200
        untouched_before = persona_db.get_persona_session("sess-untouched", user_id="1", include_deleted=False)
        assert untouched_before is not None
        assert client.post("/api/v1/persona/live/sessions/sess-b/focus").status_code == 200

    untouched_after = persona_db.get_persona_session("sess-untouched", user_id="1", include_deleted=False)
    assert untouched_after is not None
    assert untouched_after["version"] == untouched_before["version"]
    assert untouched_after["last_modified"] == untouched_before["last_modified"]


def test_live_session_focus_does_not_paginate_all_sessions(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    monkeypatch.setattr(live_control_module, "_SESSION_SCAN_PAGE_SIZE", 2)
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    for idx in range(5):
        _create_session(persona_db, user_id="1", persona_id="persona_a", session_id=f"sess-{idx}")

    original_list = persona_db.list_persona_sessions

    def fail_on_scan_page(*args, **kwargs):
        if int(kwargs.get("offset") or 0) > 0:
            raise AssertionError("focus should not paginate every persona session")
        return original_list(*args, **kwargs)

    monkeypatch.setattr(persona_db, "list_persona_sessions", fail_on_scan_page)

    with _client_for_user(1, persona_db) as client:
        response = client.post("/api/v1/persona/live/sessions/sess-4/focus")

    assert response.status_code == 200
    assert response.json()["session"]["session_id"] == "sess-4"


def test_live_session_list_reports_focus_outside_returned_page(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    focused_session_id = "sess-old-focused"
    _create_session(
        persona_db,
        user_id="1",
        persona_id="persona_a",
        session_id=focused_session_id,
        preferences={
            "persona_live_control": {
                "focus": {
                    "focused": True,
                    "focused_at": "2026-05-20T00:00:00+00:00",
                    "focus_generation": 999,
                }
            }
        },
    )
    for idx in range(5):
        _create_session(
            persona_db,
            user_id="1",
            persona_id="persona_a",
            session_id=f"sess-newer-{idx}",
        )

    with _client_for_user(1, persona_db) as client:
        listed = client.get("/api/v1/persona/live/sessions?limit=3")

    assert listed.status_code == 200
    payload = listed.json()
    assert payload["focused_session_id"] == focused_session_id
    assert focused_session_id not in {item["session_id"] for item in payload["sessions"]}


def test_live_session_list_does_not_paginate_all_sessions_for_focus(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    monkeypatch.setattr(live_control_module, "_SESSION_SCAN_PAGE_SIZE", 2)
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    focused_session_id = "sess-old-focused"
    _create_session(
        persona_db,
        user_id="1",
        persona_id="persona_a",
        session_id=focused_session_id,
        preferences={
            "persona_live_control": {
                "focus": {
                    "focused": True,
                    "focused_at": "2026-05-20T00:00:00+00:00",
                    "focus_generation": 999,
                }
            }
        },
    )
    for idx in range(5):
        _create_session(
            persona_db,
            user_id="1",
            persona_id="persona_a",
            session_id=f"sess-newer-{idx}",
        )

    original_list = persona_db.list_persona_sessions

    def fail_on_scan_page(*args, **kwargs):
        if int(kwargs.get("offset") or 0) > 0:
            raise AssertionError("focus lookup should not paginate every persona session")
        return original_list(*args, **kwargs)

    monkeypatch.setattr(persona_db, "list_persona_sessions", fail_on_scan_page)

    with _client_for_user(1, persona_db) as client:
        listed = client.get("/api/v1/persona/live/sessions?limit=3")

    assert listed.status_code == 200
    payload = listed.json()
    assert payload["focused_session_id"] == focused_session_id
    assert focused_session_id not in {item["session_id"] for item in payload["sessions"]}


def test_persona_stream_user_message_marks_live_session_connected_and_cleanup(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    manager = SessionManager()
    persisted_turns: list[dict[str, object]] = []
    _install_persona_stream_test_stubs(monkeypatch, manager, persisted_turns=persisted_turns)
    _create_profile(persona_db, user_id="1", persona_id="research_assistant", name="Research Assistant")
    session_id = "sess-user-message-ws"
    _create_session(persona_db, user_id="1", persona_id="research_assistant", session_id=session_id)

    with _client_for_user(1, persona_db) as client:
        with client.websocket_connect("/api/v1/persona/stream") as ws:
            _ = json.loads(ws.receive_text())
            ws.send_text(
                json.dumps(
                    {
                        "type": "user_message",
                        "session_id": session_id,
                        "text": "What should I focus on next?",
                        "client_message_id": f" {'u' * 140} ",
                    }
                )
            )
            _recv_until(
                ws,
                lambda data: data.get("event") == "tool_plan" and data.get("session_id") == session_id,
            )
            listed = client.get("/api/v1/persona/live/sessions")

        disconnected = client.get("/api/v1/persona/live/sessions")

    assert listed.status_code == 200
    assert _session_summary(listed.json(), session_id)["lifecycle"] == "connected"
    assert disconnected.status_code == 200
    assert _session_summary(disconnected.json(), session_id)["lifecycle"] == "idle"
    turns = manager.list_turns(session_id=session_id, user_id="1", limit=10)
    user_turn = next(turn for turn in turns if turn["type"] == "user_message")
    assert user_turn["metadata"]["client_message_id"] == "u" * 128
    persisted_user_turn = next(turn for turn in persisted_turns if turn["turn_type"] == "user_message")
    assert persisted_user_turn["metadata"]["client_message_id"] == "u" * 128


def test_persona_stream_user_message_rejects_stopped_session(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    manager = SessionManager()
    persisted_turns: list[dict[str, object]] = []
    _install_persona_stream_test_stubs(monkeypatch, manager, persisted_turns=persisted_turns)
    monkeypatch.setattr(persona_ep, "_open_persona_ws_db", lambda _user_id: persona_db)
    _create_profile(persona_db, user_id="1", persona_id="research_assistant", name="Research Assistant")
    session_id = "sess-stopped-user-message"
    _create_session(persona_db, user_id="1", persona_id="research_assistant", session_id=session_id, status="closed")

    with _client_for_user(1, persona_db) as client:
        with client.websocket_connect("/api/v1/persona/stream") as ws:
            _ = json.loads(ws.receive_text())
            ws.send_text(
                json.dumps(
                    {
                        "type": "user_message",
                        "session_id": session_id,
                        "text": "Can you still hear me?",
                    }
                )
            )
            notice = _recv_until(
                ws,
                lambda data: data.get("event") == "notice"
                and data.get("reason_code") == "SESSION_TERMINAL"
                and data.get("session_id") == session_id,
            )

    assert notice["level"] == "error"
    assert manager.list_turns(session_id=session_id, user_id="1", limit=10) == []
    assert persisted_turns == []


def test_persona_stream_voice_commit_rejects_stopped_session_before_commit_side_effects(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    manager = SessionManager()
    persisted_turns: list[dict[str, object]] = []
    _install_persona_stream_test_stubs(monkeypatch, manager, persisted_turns=persisted_turns)
    monkeypatch.setattr(persona_ep, "_open_persona_ws_db", lambda _user_id: persona_db)
    _create_profile(persona_db, user_id="1", persona_id="research_assistant", name="Research Assistant")
    session_id = "sess-stopped-voice-commit"
    _create_session(persona_db, user_id="1", persona_id="research_assistant", session_id=session_id, status="closed")

    with _client_for_user(1, persona_db) as client:
        with client.websocket_connect("/api/v1/persona/stream") as ws:
            _ = json.loads(ws.receive_text())
            ws.send_text(
                json.dumps(
                    {
                        "type": "voice_commit",
                        "session_id": session_id,
                        "transcript": "This should not commit.",
                    }
                )
            )
            notice = _recv_until(
                ws,
                lambda data: data.get("event") == "notice"
                and data.get("reason_code") == "SESSION_TERMINAL"
                and data.get("session_id") == session_id,
            )

    assert notice["level"] == "error"
    assert not any(turn.get("turn_type") == "voice_commit" for turn in persisted_turns)
    preferences = manager.get_preferences(session_id=session_id, user_id="1")
    assert preferences.get("last_turn_type") != "voice_commit"


def test_persona_stream_user_message_preserves_private_live_control_preferences(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    manager = SessionManager()
    _install_persona_stream_test_stubs(monkeypatch, manager)
    monkeypatch.setattr(persona_ep, "_open_persona_ws_db", lambda _user_id: persona_db)
    _create_profile(persona_db, user_id="1", persona_id="research_assistant", name="Research Assistant")

    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/live/sessions",
            json={
                "persona_id": "research_assistant",
                "reuse_policy": "create_new",
                "idempotency_key": "preserve-private-key",
            },
        )
        assert created.status_code == 200
        session_id = created.json()["session"]["session_id"]

        with client.websocket_connect("/api/v1/persona/stream") as ws:
            _ = json.loads(ws.receive_text())
            ws.send_text(
                json.dumps(
                    {
                        "type": "user_message",
                        "session_id": session_id,
                        "text": "Please continue.",
                    }
                )
            )
            _recv_until(
                ws,
                lambda data: data.get("event") == "tool_plan" and data.get("session_id") == session_id,
            )

    row = persona_db.get_persona_session(session_id, user_id="1", include_deleted=False)
    assert row is not None
    live_preferences = row["preferences"]["persona_live_control"]
    assert live_preferences["create_idempotency_key"] == "preserve-private-key"
    assert live_preferences["focus"]["focused"] is True


def test_persona_stream_voice_config_marks_live_session_connected_and_cleanup(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    manager = SessionManager()
    _install_persona_stream_test_stubs(monkeypatch, manager)
    _create_profile(persona_db, user_id="1", persona_id="research_assistant", name="Research Assistant")
    session_id = "sess-voice-config-ws"
    _create_session(persona_db, user_id="1", persona_id="research_assistant", session_id=session_id)

    with _client_for_user(1, persona_db) as client:
        with client.websocket_connect("/api/v1/persona/stream") as ws:
            _ = json.loads(ws.receive_text())
            ws.send_text(
                json.dumps(
                    {
                        "type": "voice_config",
                        "session_id": session_id,
                        "voice": {"trigger_phrases": []},
                    }
                )
            )
            _recv_until(
                ws,
                lambda data: data.get("event") == "notice"
                and data.get("reason_code") == "VOICE_CONFIG_UPDATED"
                and data.get("session_id") == session_id,
            )
            listed = client.get("/api/v1/persona/live/sessions")

        disconnected = client.get("/api/v1/persona/live/sessions")

    assert listed.status_code == 200
    assert _session_summary(listed.json(), session_id)["lifecycle"] == "connected"
    assert disconnected.status_code == 200
    assert _session_summary(disconnected.json(), session_id)["lifecycle"] == "idle"


def test_persona_stream_presence_refcounts_multiple_websockets_on_same_session(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    manager = SessionManager()
    _install_persona_stream_test_stubs(monkeypatch, manager)
    _create_profile(persona_db, user_id="1", persona_id="research_assistant", name="Research Assistant")
    session_id = "sess-two-websockets"
    _create_session(persona_db, user_id="1", persona_id="research_assistant", session_id=session_id)

    with _client_for_user(1, persona_db) as client:
        with client.websocket_connect("/api/v1/persona/stream") as ws1:
            _ = json.loads(ws1.receive_text())
            ws1.send_text(
                json.dumps(
                    {
                        "type": "voice_config",
                        "session_id": session_id,
                        "voice": {"trigger_phrases": []},
                    }
                )
            )
            _recv_until(
                ws1,
                lambda data: data.get("event") == "notice"
                and data.get("reason_code") == "VOICE_CONFIG_UPDATED"
                and data.get("session_id") == session_id,
            )
            with client.websocket_connect("/api/v1/persona/stream") as ws2:
                _ = json.loads(ws2.receive_text())
                ws2.send_text(
                    json.dumps(
                        {
                            "type": "voice_config",
                            "session_id": session_id,
                            "voice": {"trigger_phrases": []},
                        }
                    )
                )
                _recv_until(
                    ws2,
                    lambda data: data.get("event") == "notice"
                    and data.get("reason_code") == "VOICE_CONFIG_UPDATED"
                    and data.get("session_id") == session_id,
                )

            after_one_close = client.get("/api/v1/persona/live/sessions")

        after_both_close = client.get("/api/v1/persona/live/sessions")

    assert after_one_close.status_code == 200
    assert _session_summary(after_one_close.json(), session_id)["lifecycle"] == "connected"
    assert after_both_close.status_code == 200
    assert _session_summary(after_both_close.json(), session_id)["lifecycle"] == "idle"


def test_live_session_terminal_status_excludes_send_text_action(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-closed", status="closed")

    with _client_for_user(1, persona_db) as client:
        listed = client.get("/api/v1/persona/live/sessions")

    summary = listed.json()["sessions"][0]
    assert summary["lifecycle"] == "stopped"
    assert summary["allowed_actions"] == []


def test_live_session_summary_redacts_sensitive_preferences(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    _create_session(
        persona_db,
        user_id="1",
        persona_id="persona_a",
        session_id="sess-sensitive",
        preferences={
            "provider_api_key": "sk-secret",
            "raw_prompt": "private instructions",
            "persona_live_control": {"create_idempotency_key": "safe-key"},
        },
    )

    with _client_for_user(1, persona_db) as client:
        listed = client.get("/api/v1/persona/live/sessions")

    encoded = str(listed.json())
    assert "preferences" not in listed.json()["sessions"][0]
    assert "sk-secret" not in encoded
    assert "private instructions" not in encoded
    assert "safe-key" not in encoded
