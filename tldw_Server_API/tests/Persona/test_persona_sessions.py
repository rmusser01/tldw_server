import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Persona.session_materialization import materialize_persona_session
from tldw_Server_API.app.core.Persona.session_manager import SessionManager


pytestmark = pytest.mark.unit

fastapi_app = FastAPI()
fastapi_app.include_router(persona_ep.router, prefix="/api/v1/persona")


@pytest.fixture()
def persona_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "persona_sessions.db"), client_id="persona-sessions-tests")
    yield db
    db.close_connection()


def _client_for_user(user_id: int, db: CharactersRAGDB):
    async def override_user():
        return User(id=user_id, username=f"user-{user_id}", email=None, is_active=True)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    return TestClient(fastapi_app)


def test_persona_sessions_requires_auth():
    with TestClient(fastapi_app) as client:
        r_list = client.get("/api/v1/persona/sessions")
        r_detail = client.get("/api/v1/persona/sessions/sess_missing")
        assert r_list.status_code == 401
        assert r_detail.status_code == 401


def test_persona_sessions_list_and_detail_roundtrip(monkeypatch, persona_db: CharactersRAGDB):
    manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)

    with _client_for_user(1, persona_db) as client:
        created = client.post("/api/v1/persona/session", json={"persona_id": "research_assistant"})
        assert created.status_code == 200
        session_id = created.json()["session_id"]

        manager.append_turn(
            session_id=session_id,
            user_id="1",
            persona_id="research_assistant",
            role="user",
            content="hello",
            turn_type="user_message",
        )

        listed = client.get("/api/v1/persona/sessions")
        assert listed.status_code == 200
        payload = listed.json()
        assert isinstance(payload, list)
        assert any(item["session_id"] == session_id for item in payload)

        detail = client.get(f"/api/v1/persona/sessions/{session_id}?limit_turns=10")
        assert detail.status_code == 200
        detail_payload = detail.json()
        assert detail_payload["session_id"] == session_id
        assert detail_payload["turn_count"] >= 1
        assert len(detail_payload["turns"]) >= 1

    fastapi_app.dependency_overrides.clear()


def test_persona_sessions_list_and_detail_fall_back_to_persisted_preferences(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)

    with _client_for_user(1, persona_db) as client:
        created = client.post("/api/v1/persona/session", json={"persona_id": "research_assistant"})
        assert created.status_code == 200
        session_id = created.json()["session_id"]

    session_row = persona_db.get_persona_session(session_id, user_id="1", include_deleted=False)
    assert session_row is not None
    assert persona_db.update_persona_session(
        session_id=session_id,
        user_id="1",
        update_data={
            "preferences_json": {
                "use_memory_context": True,
                "use_companion_context": False,
                "use_persona_state_context": False,
                "memory_top_k": 3,
            }
        },
        expected_version=int(session_row["version"]),
    )

    restarted_manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: restarted_manager)

    with _client_for_user(1, persona_db) as client:
        listed = client.get("/api/v1/persona/sessions")
        assert listed.status_code == 200
        payload = listed.json()
        matched = next(item for item in payload if item["session_id"] == session_id)
        assert matched["preferences"]["use_companion_context"] is False
        assert matched["preferences"]["use_persona_state_context"] is False
        assert matched["preferences"]["memory_top_k"] == 3

        detail = client.get(f"/api/v1/persona/sessions/{session_id}?limit_turns=10")
        assert detail.status_code == 200
        detail_payload = detail.json()
        assert detail_payload["preferences"]["use_companion_context"] is False
        assert detail_payload["preferences"]["use_persona_state_context"] is False
        assert detail_payload["preferences"]["memory_top_k"] == 3

    fastapi_app.dependency_overrides.clear()


def test_persona_sessions_list_filters_by_surface(monkeypatch, persona_db: CharactersRAGDB):
    manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)

    with _client_for_user(1, persona_db) as client:
        generic = client.post("/api/v1/persona/session", json={"persona_id": "research_assistant"})
        assert generic.status_code == 200
        generic_session_id = generic.json()["session_id"]

        companion = client.post(
            "/api/v1/persona/session",
            json={
                "persona_id": "research_assistant",
                "surface": "companion.conversation",
            },
        )
        assert companion.status_code == 200
        companion_session_id = companion.json()["session_id"]

        filtered = client.get(
            "/api/v1/persona/sessions?persona_id=research_assistant&surface=companion.conversation"
        )
        assert filtered.status_code == 200
        payload = filtered.json()
        assert [item["session_id"] for item in payload] == [companion_session_id]
        assert companion_session_id in {item["session_id"] for item in payload}
        assert generic_session_id not in {item["session_id"] for item in payload}

    fastapi_app.dependency_overrides.clear()


def test_persona_sessions_created_with_surface_do_not_persist_surface_preference(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)

    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/session",
            json={"persona_id": "research_assistant", "surface": "companion.conversation"},
        )
        assert created.status_code == 200
        session_id = created.json()["session_id"]

        active_list = client.get("/api/v1/persona/sessions?surface=companion.conversation")
        active_detail = client.get(f"/api/v1/persona/sessions/{session_id}")

    assert active_list.status_code == 200
    active_matched = next(item for item in active_list.json() if item["session_id"] == session_id)
    assert "companion_activity_surface" not in active_matched["preferences"]
    assert active_detail.status_code == 200
    assert "companion_activity_surface" not in active_detail.json()["preferences"]

    row = persona_db.get_persona_session(session_id, user_id="1", include_deleted=False)
    assert row is not None
    assert row["activity_surface"] == "companion.conversation"
    assert "companion_activity_surface" not in row["preferences"]

    restarted_manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: restarted_manager)
    with _client_for_user(1, persona_db) as client:
        listed = client.get("/api/v1/persona/sessions?surface=companion.conversation")

    assert listed.status_code == 200
    matched = next(item for item in listed.json() if item["session_id"] == session_id)
    assert "companion_activity_surface" not in matched["preferences"]

    fastapi_app.dependency_overrides.clear()


def test_persona_sessions_redact_live_control_preferences(
    monkeypatch,
    persona_db: CharactersRAGDB,
):
    manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)

    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/live/sessions",
            json={
                "persona_id": "research_assistant",
                "reuse_policy": "create_new",
                "idempotency_key": "sensitive-create-key",
                "surface": "companion.conversation",
            },
        )
        assert created.status_code == 200
        session_id = created.json()["session"]["session_id"]

        listed = client.get("/api/v1/persona/sessions?surface=companion.conversation")
        detail = client.get(f"/api/v1/persona/sessions/{session_id}")

    assert listed.status_code == 200
    matched = next(item for item in listed.json() if item["session_id"] == session_id)
    assert "persona_live_control" not in matched["preferences"]
    assert "sensitive-create-key" not in str(matched)
    assert "companion_activity_surface" not in matched["preferences"]

    assert detail.status_code == 200
    detail_payload = detail.json()
    assert "persona_live_control" not in detail_payload["preferences"]
    assert "sensitive-create-key" not in str(detail_payload)
    assert "companion_activity_surface" not in detail_payload["preferences"]

    fastapi_app.dependency_overrides.clear()


def test_materialized_session_policy_rules_are_normalized(persona_db: CharactersRAGDB):
    manager = SessionManager()
    _ = persona_db.create_persona_profile(
        {
            "id": "persona_policy",
            "user_id": "1",
            "name": "Policy Persona",
            "mode": "session_scoped",
            "system_prompt": "Policy helper",
            "is_active": True,
        }
    )
    _ = persona_db.create_persona_session(
        {
            "id": "sess_policy_rules",
            "persona_id": "persona_policy",
            "user_id": "1",
            "mode": "session_scoped",
            "status": "active",
            "scope_snapshot_json": {},
            "preferences_json": {
                "session_policy_rules": [
                    {
                        "rule_kind": "MCP_TOOL",
                        "rule_name": "Notes.Create",
                        "allowed": False,
                        "require_confirmation": 1,
                    },
                    {"rule_kind": "invalid", "rule_name": "ignored", "allowed": True},
                ]
            },
        }
    )

    materialized = materialize_persona_session(
        persona_db,
        session_manager=manager,
        user_id="1",
        persona_id="persona_policy",
        resume_session_id="sess_policy_rules",
    )

    assert materialized.session_id == "sess_policy_rules"
    assert manager.get_preferences(session_id="sess_policy_rules", user_id="1")["session_policy_rules"] == [
        {
            "rule_kind": "mcp_tool",
            "rule_name": "notes.create",
            "allowed": False,
            "require_confirmation": True,
        }
    ]


def test_persona_session_resume_rejects_ownership_mismatch(monkeypatch, persona_db: CharactersRAGDB):
    manager = SessionManager()
    _ = manager.create(user_id="1", persona_id="research_assistant", resume_session_id="sess_owned_by_1")
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)

    with _client_for_user(2, persona_db) as client:
        resp = client.post(
            "/api/v1/persona/session",
            json={
                "persona_id": "research_assistant",
                "resume_session_id": "sess_owned_by_1",
            },
        )
        assert resp.status_code == 403

    fastapi_app.dependency_overrides.clear()


def test_persona_session_resume_rejects_persona_mismatch(monkeypatch, persona_db: CharactersRAGDB):
    manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)

    _ = persona_db.create_persona_profile(
        {
            "id": "persona_a",
            "user_id": "1",
            "name": "Persona A",
            "mode": "session_scoped",
            "system_prompt": "A",
            "is_active": True,
        }
    )
    _ = persona_db.create_persona_profile(
        {
            "id": "persona_b",
            "user_id": "1",
            "name": "Persona B",
            "mode": "session_scoped",
            "system_prompt": "B",
            "is_active": True,
        }
    )
    _ = persona_db.create_persona_session(
        {
            "id": "sess_bound_persona_a",
            "persona_id": "persona_a",
            "user_id": "1",
            "mode": "session_scoped",
            "reuse_allowed": False,
            "status": "active",
            "scope_snapshot_json": {},
        }
    )

    with _client_for_user(1, persona_db) as client:
        resp = client.post(
            "/api/v1/persona/session",
            json={
                "persona_id": "persona_b",
                "resume_session_id": "sess_bound_persona_a",
            },
        )
        assert resp.status_code == 409
        assert "different persona_id" in str(resp.json().get("detail"))
        assert manager.get("sess_bound_persona_a") is None

    fastapi_app.dependency_overrides.clear()


def test_persona_session_detail_is_user_scoped(monkeypatch, persona_db: CharactersRAGDB):
    manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)
    persona_id = persona_db.create_persona_profile(
        {
            "id": "research_assistant",
            "user_id": "1",
            "name": "Research Assistant",
            "mode": "session_scoped",
            "system_prompt": "Helper",
            "is_active": True,
        }
    )
    _ = persona_db.create_persona_session(
        {
            "id": "sess_scoped",
            "persona_id": persona_id,
            "user_id": "1",
            "mode": "session_scoped",
            "reuse_allowed": False,
            "status": "active",
            "scope_snapshot_json": {},
        }
    )

    with _client_for_user(2, persona_db) as client:
        resp = client.get("/api/v1/persona/sessions/sess_scoped")
        assert resp.status_code == 404

    fastapi_app.dependency_overrides.clear()


def test_persona_sessions_return_404_when_disabled(monkeypatch, persona_db: CharactersRAGDB):
    manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)
    monkeypatch.setattr(persona_ep, "is_persona_enabled", lambda: False)

    with _client_for_user(1, persona_db) as client:
        r_list = client.get("/api/v1/persona/sessions")
        r_detail = client.get("/api/v1/persona/sessions/sess_disabled")
        assert r_list.status_code == 404
        assert r_detail.status_code == 404

    fastapi_app.dependency_overrides.clear()
