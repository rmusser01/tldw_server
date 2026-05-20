import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
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


def test_live_session_create_uses_existing_session_materialization(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())

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
    assert row["preferences"]["companion_activity_surface"] == "companion.conversation"
    assert row["activity_surface"] == "companion.conversation"


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


def test_live_session_focus_rejects_other_user_session(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-owned")

    with _client_for_user(2, persona_db) as client:
        resp = client.post("/api/v1/persona/live/sessions/sess-owned/focus")

    assert resp.status_code == 403


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


def test_live_session_active_stream_presence_is_connected(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: SessionManager())
    _create_profile(persona_db, user_id="1", persona_id="persona_a")
    _create_session(persona_db, user_id="1", persona_id="persona_a", session_id="sess-a")
    persona_live_stream_registry.mark_connected(user_id="1", session_id="sess-a")

    with _client_for_user(1, persona_db) as client:
        listed = client.get("/api/v1/persona/live/sessions")

    assert listed.status_code == 200
    assert listed.json()["sessions"][0]["lifecycle"] == "connected"


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
