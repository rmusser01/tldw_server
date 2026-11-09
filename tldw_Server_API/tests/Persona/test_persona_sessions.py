import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core import config as config_module
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Persona.session_manager import SessionManager
from tldw_Server_API.app.core.Persona.session_materialization import materialize_persona_session

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


def test_legacy_runtime_transcript_preserves_detail_and_export_limit(monkeypatch, persona_db):
    manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)
    monkeypatch.setattr(persona_ep, "_get_persona_rbac_flags", lambda: (True, False))
    try:
        with _client_for_user(1, persona_db) as client:
            created = client.post("/api/v1/persona/session", json={"persona_id": "research_assistant"})
            session_id = created.json()["session_id"]
            turns = [manager.append_turn(session_id=session_id, user_id="1", persona_id="research_assistant",
                                         role="user", content=str(index)) for index in range(3)]
            detail = client.get(f"/api/v1/persona/sessions/{session_id}?limit_turns=1")
            exported = client.get(f"/api/v1/persona/sessions/{session_id}/export?limit_turns=1")
            assert detail.status_code == exported.status_code == 200  # nosec B101
            assert detail.json()["turn_count"] == 3  # nosec B101
            assert detail.json()["turns"] == turns[-1:]  # nosec B101
            assert [turn["turn_id"] for turn in exported.json()["turns"]] == [turns[-1]["turn_id"]]  # nosec B101
    finally:
        fastapi_app.dependency_overrides.clear()


def test_materialized_session_reads_memory_top_k_from_settings_attributes(monkeypatch, persona_db: CharactersRAGDB):
    class AttributeSettings:
        PERSONA_MEMORY_TOP_K = "6"

    manager = SessionManager()
    monkeypatch.setattr(config_module, "settings", AttributeSettings())

    materialized = materialize_persona_session(
        persona_db,
        session_manager=manager,
        user_id="1",
        persona_id="research_assistant",
    )

    row = persona_db.get_persona_session(materialized.session_id, user_id="1", include_deleted=False)
    assert row is not None
    assert row["preferences"]["memory_top_k"] == 6


def test_persona_session_export_returns_selected_session_transcript(monkeypatch, persona_db: CharactersRAGDB):
    manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)
    monkeypatch.setattr(persona_ep, "_get_persona_rbac_flags", lambda: (True, False))

    with _client_for_user(1, persona_db) as client:
        created = client.post("/api/v1/persona/session", json={"persona_id": "research_assistant"})
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        sensitive_value = "sensitive" + "-value"

        manager.append_turn(
            session_id=session_id,
            user_id="1",
            persona_id="research_assistant",
            role="user",
            content="hello",
            turn_type="user_message",
            metadata={
                "visible": "kept",
                "optional_value": None,
                "list_with_null": ["first", None, "last"],
                "nested": {"kept_null": None},
                "bytes_base64": "raw-audio",
                "system_prompt": "hidden prompt",
                "auth_token": sensitive_value,
            },
        )
        manager.append_turn(
            session_id=session_id,
            user_id="1",
            persona_id="research_assistant",
            role="assistant",
            content="Hi there.",
            turn_type="assistant_message",
        )

        exported = client.get(f"/api/v1/persona/sessions/{session_id}/export")
        assert exported.status_code == 200
        payload = exported.json()
        assert payload["session_id"] == session_id
        assert payload["persona_id"] == "research_assistant"
        assert payload["format"] == "json"
        assert payload["turn_count"] == 2
        assert payload["redaction_markers"] == [
            "metadata.auth_token",
            "metadata.bytes_base64",
            "metadata.system_prompt",
        ]
        assert [turn["event_type"] for turn in payload["turns"]] == ["user_message", "assistant_message"]
        assert payload["turns"][0]["content"] == "hello"
        assert payload["turns"][0]["metadata"] == {
            "list_with_null": ["first", None, "last"],
            "nested": {"kept_null": None},
            "optional_value": None,
            "visible": "kept",
        }
        assert sensitive_value not in str(payload)
        assert "hidden prompt" not in str(payload)
        assert "raw-audio" not in str(payload)

    fastapi_app.dependency_overrides.clear()


def test_persona_session_export_rejects_when_export_disabled(monkeypatch, persona_db: CharactersRAGDB):
    monkeypatch.setattr(persona_ep, "_get_persona_rbac_flags", lambda: (False, False))

    with _client_for_user(1, persona_db) as client:
        resp = client.get("/api/v1/persona/sessions/sess_export_disabled/export")
        assert resp.status_code == 403
        assert "export is disabled" in resp.json()["detail"]

    fastapi_app.dependency_overrides.clear()


def test_persona_session_export_redacts_non_object_top_level_metadata():
    markers: set[str] = set()

    redacted = persona_ep._redacted_persona_export_metadata(
        ["unexpected", {"auth_token": "hidden"}],
        markers=markers,
    )

    assert redacted == {}
    assert markers == {"metadata"}


def test_persona_session_response_helpers_normalize_invalid_runtime_mode():
    row = {
        "id": "sess_invalid_mode",
        "persona_id": "research_assistant",
        "mode": "unexpected_mode",
        "status": "active",
        "reuse_allowed": False,
        "scope_snapshot": {},
    }

    assert persona_ep._persona_session_summary_from_db(row).runtime_mode == "session_scoped"
    assert persona_ep._persona_session_detail_from_db(row).runtime_mode == "session_scoped"


def test_persona_session_export_is_user_scoped(monkeypatch, persona_db: CharactersRAGDB):
    manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)
    monkeypatch.setattr(persona_ep, "_get_persona_rbac_flags", lambda: (True, False))
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
            "id": "sess_export_scoped",
            "persona_id": persona_id,
            "user_id": "1",
            "mode": "session_scoped",
            "reuse_allowed": False,
            "status": "active",
            "scope_snapshot_json": {},
        }
    )

    with _client_for_user(2, persona_db) as client:
        resp = client.get("/api/v1/persona/sessions/sess_export_scoped/export")
        assert resp.status_code == 404

    fastapi_app.dependency_overrides.clear()


def test_persona_session_export_refuses_unrecoverable_legacy_snapshot(monkeypatch, persona_db: CharactersRAGDB):
    manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)
    monkeypatch.setattr(persona_ep, "_get_persona_rbac_flags", lambda: (True, False))
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
            "id": "sess_export_not_live",
            "persona_id": persona_id,
            "user_id": "1",
            "mode": "session_scoped",
            "reuse_allowed": False,
            "status": "active",
            "scope_snapshot_json": {},
        }
    )

    with _client_for_user(1, persona_db) as client:
        resp = client.get("/api/v1/persona/sessions/sess_export_not_live/export")
        assert resp.status_code == 409
        assert "No retained transcript" in resp.json()["detail"]  # nosec B101 -- Assert truthful refusal for a legacy session with no retained transcript.

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

    row = persona_db.get_persona_session(session_id, user_id="1", include_deleted=False)
    assert row is not None
    preferences = dict(row["preferences"])
    preferences["custom_visible_preference"] = {"retained": True}
    preferences["provider_api_key"] = "sk-private"
    preferences["raw_prompt"] = "private raw prompt"
    assert persona_db.update_persona_session(
        session_id=session_id,
        user_id="1",
        update_data={"preferences_json": preferences},
        expected_version=int(row["version"]),
    )

    with _client_for_user(1, persona_db) as client:
        listed = client.get("/api/v1/persona/sessions?surface=companion.conversation")
        detail = client.get(f"/api/v1/persona/sessions/{session_id}")

    assert listed.status_code == 200
    matched = next(item for item in listed.json() if item["session_id"] == session_id)
    assert "persona_live_control" not in matched["preferences"]
    assert "sensitive-create-key" not in str(matched)
    assert "companion_activity_surface" not in matched["preferences"]
    assert matched["preferences"]["custom_visible_preference"] == {"retained": True}
    assert "provider_api_key" not in matched["preferences"]
    assert "raw_prompt" not in matched["preferences"]

    assert detail.status_code == 200
    detail_payload = detail.json()
    assert "persona_live_control" not in detail_payload["preferences"]
    assert "sensitive-create-key" not in str(detail_payload)
    assert "companion_activity_surface" not in detail_payload["preferences"]
    assert detail_payload["preferences"]["custom_visible_preference"] == {"retained": True}
    assert "provider_api_key" not in detail_payload["preferences"]
    assert "raw_prompt" not in detail_payload["preferences"]

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


@pytest.fixture
def pending_plan_session(monkeypatch, persona_db, request):
    manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: manager)
    monkeypatch.setattr(persona_ep, "_get_persona_rbac_flags", lambda: (True, False))
    try:
        with _client_for_user(1, persona_db) as client:
            created = client.post("/api/v1/persona/session", json={"persona_id": "research_assistant"})
            assert created.status_code == 200
            sid = created.json()["session_id"]
        manager.put_plan(
            session_id=sid,
            user_id="1",
            persona_id="research_assistant",
            plan_id="pending",
            steps=[{"idx": 0, "tool": "rag_search", "args": {"query": "review me"}}],
            requires_persisted_session=True,
        )
        session_status = getattr(request, "param", "active")
        if session_status != "active":
            row = persona_db.get_persona_session(sid, user_id="1")
            persona_db.update_persona_session(
                session_id=sid, user_id="1", expected_version=row["version"], update_data={"status": session_status}
            )
        yield sid, manager
    finally:
        fastapi_app.dependency_overrides.clear()


def test_session_detail_hydrates_owned_pending_plan(persona_db, pending_plan_session):
    sid, _ = pending_plan_session
    with _client_for_user(1, persona_db) as client:
        detail = client.get(f"/api/v1/persona/sessions/{sid}")
    assert detail.status_code == 200
    assert detail.json()["pending_plan"]["plan_id"] == "pending"
    assert detail.json()["pending_plan"]["steps"][0]["args"] == {"query": "review me"}


@pytest.mark.parametrize("pending_plan_session", ["closed", "archived"], indirect=True)
def test_session_detail_excludes_terminal_pending_plan(persona_db, pending_plan_session):
    sid, _ = pending_plan_session
    with _client_for_user(1, persona_db) as client:
        detail = client.get(f"/api/v1/persona/sessions/{sid}")
    assert detail.status_code == 200
    assert detail.json()["pending_plan"] is None


@pytest.mark.parametrize("pending_plan_session", ["active", "closed", "archived"], indirect=True)
def test_session_list_excludes_pending_plans(persona_db, pending_plan_session):
    sid, _ = pending_plan_session
    with _client_for_user(1, persona_db) as client:
        listed = client.get("/api/v1/persona/sessions")
    assert listed.status_code == 200
    assert any(item["session_id"] == sid for item in listed.json())
    assert all("pending_plan" not in item for item in listed.json())


def test_session_export_excludes_pending_plan(persona_db, pending_plan_session):
    sid, _ = pending_plan_session
    with _client_for_user(1, persona_db) as client:
        exported = client.get(f"/api/v1/persona/sessions/{sid}/export")
    assert exported.status_code == 200
    assert "pending_plan" not in exported.json()


@pytest.mark.parametrize("pending_plan_session", ["active", "closed", "archived"], indirect=True)
def test_session_detail_does_not_consume_pending_plan(persona_db, pending_plan_session):
    sid, manager = pending_plan_session
    pending = manager.get_plan(session_id=sid, user_id="1", plan_id="pending")
    with _client_for_user(1, persona_db) as client:
        assert client.get(f"/api/v1/persona/sessions/{sid}").status_code == 200
    assert manager.get_plan(session_id=sid, user_id="1", plan_id="pending") is pending


@pytest.mark.parametrize("pending_plan_session", ["active", "closed", "archived"], indirect=True)
def test_session_detail_with_pending_plan_rejects_other_owner(persona_db, pending_plan_session):
    sid, _ = pending_plan_session
    with _client_for_user(2, persona_db) as other:
        assert other.get(f"/api/v1/persona/sessions/{sid}").status_code == 404


def test_cold_foreign_resume_cannot_occupy_owner_session(monkeypatch, persona_db, tmp_path):
    """A cold process must preserve the legitimate owner's separate SQLite session."""
    owner = materialize_persona_session(
        persona_db,
        session_manager=SessionManager(),
        user_id="1",
        persona_id="research_assistant",
    )
    session_id = owner.session_id
    original = persona_db.get_persona_session(session_id, user_id="1", include_deleted=False)
    other_db = CharactersRAGDB(str(tmp_path / "other_user.db"), client_id="cold-resume-other")
    cold_manager = SessionManager()
    monkeypatch.setattr(persona_ep, "get_session_manager", lambda: cold_manager)
    try:
        with _client_for_user(2, other_db) as client:
            refused = client.post(
                "/api/v1/persona/session",
                json={
                    "persona_id": "research_assistant",
                    "resume_session_id": session_id,
                },
            )
        assert refused.status_code == 403  # nosec B101 -- A foreign cold ID cannot become an owned session.
        assert other_db.get_persona_session(session_id, user_id="2", include_deleted=False) is None  # nosec B101
        assert cold_manager.get(session_id) is None  # nosec B101 -- Refusal must precede global manager occupation.
        with _client_for_user(1, persona_db) as client:
            resumed = client.post(
                "/api/v1/persona/session",
                json={
                    "persona_id": "research_assistant",
                    "resume_session_id": session_id,
                },
            )
        assert resumed.status_code == 200 and resumed.json()["session_id"] == session_id  # nosec B101
        assert persona_db.get_persona_session(session_id, user_id="1", include_deleted=False) == original  # nosec B101
    finally:
        fastapi_app.dependency_overrides.clear()
        other_db.close_connection()


@pytest.mark.parametrize("saved_runtime", [False, True])
def test_resume_without_owned_parent_cannot_create_from_supplied_id(persona_db, saved_runtime):
    manager = SessionManager()
    session_id = "unowned-cold-resume-id"
    if saved_runtime:
        manager.create(user_id="1", persona_id="research_assistant", resume_session_id=session_id, persisted=True)
    with pytest.raises(PermissionError):
        materialize_persona_session(
            persona_db,
            session_manager=manager,
            user_id="1",
            persona_id="research_assistant",
            resume_session_id=session_id,
        )
    assert persona_db.get_persona_session(session_id, user_id="1", include_deleted=False) is None  # nosec B101


def test_existing_owned_unpersisted_scaffold_can_materialize(persona_db):
    manager = SessionManager()
    pending = manager.create(user_id="1", persona_id="research_assistant", resume_session_id="owned-temporary-id")
    materialized = materialize_persona_session(
        persona_db,
        session_manager=manager,
        user_id="1",
        persona_id="research_assistant",
        resume_session_id=pending.session_id,
    )
    assert materialized.session_id == pending.session_id and materialized.created_new_session  # nosec B101
    assert manager.get(pending.session_id) is pending and pending.persisted  # nosec B101
