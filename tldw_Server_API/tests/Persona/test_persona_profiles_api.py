import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.tests.Characters.test_character_functionality_db import sample_card_data


pytestmark = pytest.mark.unit

fastapi_app = FastAPI()
fastapi_app.include_router(persona_ep.router, prefix="/api/v1/persona")


def _client_for_user(user_id: int, db: CharactersRAGDB):
    async def override_user():
        return User(id=user_id, username=f"persona-user-{user_id}", email=None, is_active=True)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    return TestClient(fastapi_app)


@pytest.fixture()
def persona_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "persona_profiles_api.db"), client_id="persona-profiles-api-tests")
    yield db
    db.close_connection()


def test_persona_profile_scope_policy_crud(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={
                "name": "Ops Assistant",
                "mode": "persistent_scoped",
                "system_prompt": "Focus on support workflows.",
            },
        )
        assert created.status_code == 201, created.text
        profile = created.json()
        persona_id = profile["id"]
        assert profile["mode"] == "persistent_scoped"
        assert profile["use_persona_state_context_default"] is True
        buddy_summary = profile["buddy_summary"]
        assert buddy_summary is not None
        assert buddy_summary["has_buddy"] is True
        assert buddy_summary["persona_name"] == "Ops Assistant"
        assert buddy_summary["role_summary"]
        assert buddy_summary["visual"]["species_id"]

        fetched = client.get(f"/api/v1/persona/profiles/{persona_id}")
        assert fetched.status_code == 200
        fetched_payload = fetched.json()
        assert fetched_payload["name"] == "Ops Assistant"
        fetched_buddy_summary = fetched_payload["buddy_summary"]
        assert fetched_buddy_summary is not None
        assert fetched_buddy_summary["persona_name"] == "Ops Assistant"
        assert fetched_buddy_summary["has_buddy"] is True

        scope_replace = client.put(
            f"/api/v1/persona/profiles/{persona_id}/scope-rules",
            json={
                "rules": [
                    {"rule_type": "conversation_id", "rule_value": "conv-001", "include": True},
                    {"rule_type": "media_tag", "rule_value": "priority", "include": True},
                    {"rule_type": "media_id", "rule_value": "101", "include": False},
                ]
            },
        )
        assert scope_replace.status_code == 200, scope_replace.text
        scope_payload = scope_replace.json()
        assert scope_payload["replaced_count"] == 3
        assert len(scope_payload["rules"]) == 3

        policy_replace = client.put(
            f"/api/v1/persona/profiles/{persona_id}/policy-rules",
            json={
                "rules": [
                    {"rule_kind": "mcp_tool", "rule_name": "media.search", "allowed": True},
                    {
                        "rule_kind": "skill",
                        "rule_name": "workspace.digest",
                        "allowed": True,
                        "require_confirmation": False,
                        "max_calls_per_turn": 2,
                    },
                ]
            },
        )
        assert policy_replace.status_code == 200, policy_replace.text
        policy_payload = policy_replace.json()
        assert policy_payload["replaced_count"] == 2
        assert len(policy_payload["rules"]) == 2

        patched = client.patch(
            f"/api/v1/persona/profiles/{persona_id}",
            json={
                "mode": "session_scoped",
                "is_active": True,
                "use_persona_state_context_default": False,
            },
        )
        assert patched.status_code == 200, patched.text
        assert patched.json()["mode"] == "session_scoped"
        assert patched.json()["use_persona_state_context_default"] is False

        listed = client.get("/api/v1/persona/profiles")
        assert listed.status_code == 200
        listed_payload = listed.json()
        listed_ids = {item["id"] for item in listed_payload}
        assert persona_id in listed_ids
        listed_item = next(item for item in listed_payload if item["id"] == persona_id)
        listed_buddy_summary = listed_item["buddy_summary"]
        assert listed_buddy_summary is not None
        assert listed_buddy_summary["has_buddy"] is True
        assert listed_buddy_summary["persona_name"] == "Ops Assistant"

        deleted = client.delete(f"/api/v1/persona/profiles/{persona_id}")
        assert deleted.status_code == 200
        assert deleted.json()["status"] == "deleted"

        missing = client.get(f"/api/v1/persona/profiles/{persona_id}")
        assert missing.status_code == 404

    fastapi_app.dependency_overrides.clear()


def test_persona_profile_projection_fails_open_when_buddy_lookup_breaks(
    persona_db: CharactersRAGDB,
    monkeypatch,
):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={"name": "Projection Recovery Persona", "mode": "session_scoped"},
        )
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]

        monkeypatch.setattr(
            persona_db,
            "get_persona_buddy",
            lambda **_kwargs: (_ for _ in ()).throw(
                CharactersRAGDBError("buddy projection exploded")
            ),
        )

        fetched = client.get(f"/api/v1/persona/profiles/{persona_id}")
        assert fetched.status_code == 200, fetched.text
        payload = fetched.json()
        assert payload["id"] == persona_id
        assert payload["buddy_summary"]["persona_name"] == "Projection Recovery Persona"

    fastapi_app.dependency_overrides.clear()


def test_create_persona_profile_sanitizes_buddy_rollback_failure(
    persona_db: CharactersRAGDB,
    monkeypatch,
):
    def _raise_buddy_failure(*args, **kwargs):
        _ = (args, kwargs)
        raise ValueError("buddy validation exploded")

    def _raise_rollback_failure(*args, **kwargs):
        _ = (args, kwargs)
        raise persona_ep.PersonaBuddyRollbackError("rollback create exploded")

    monkeypatch.setattr(
        persona_ep,
        "_ensure_persona_buddy_after_profile_mutation",
        _raise_buddy_failure,
    )
    monkeypatch.setattr(
        persona_ep,
        "_rollback_created_persona_profile_after_buddy_failure",
        _raise_rollback_failure,
    )

    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={
                "name": "Rollback Create Persona",
                "mode": "persistent_scoped",
                "system_prompt": "Trigger rollback failure.",
            },
        )

    assert created.status_code == 500, created.text
    assert created.json()["detail"] == "Failed to roll back persona profile creation"
    fastapi_app.dependency_overrides.clear()


def test_update_persona_profile_sanitizes_buddy_rollback_failure(
    persona_db: CharactersRAGDB,
    monkeypatch,
):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={
                "name": "Rollback Update Persona",
                "mode": "persistent_scoped",
                "system_prompt": "Create before update rollback failure.",
            },
        )
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]

        def _raise_buddy_failure(*args, **kwargs):
            _ = (args, kwargs)
            raise ValueError("buddy validation exploded")

        def _raise_rollback_failure(*args, **kwargs):
            _ = (args, kwargs)
            raise persona_ep.PersonaBuddyRollbackError("rollback update exploded")

        monkeypatch.setattr(
            persona_ep,
            "_ensure_persona_buddy_after_profile_mutation",
            _raise_buddy_failure,
        )
        monkeypatch.setattr(
            persona_ep,
            "_rollback_updated_persona_profile_after_buddy_failure",
            _raise_rollback_failure,
        )

        updated = client.patch(
            f"/api/v1/persona/profiles/{persona_id}",
            json={"mode": "session_scoped"},
        )

    assert updated.status_code == 500, updated.text
    assert updated.json()["detail"] == "Failed to roll back persona profile update"
    fastapi_app.dependency_overrides.clear()


def test_list_persona_profiles_uses_bulk_buddy_lookup_for_profile_collections(
    persona_db: CharactersRAGDB,
    monkeypatch,
):
    with _client_for_user(1, persona_db) as client:
        first = client.post(
            "/api/v1/persona/profiles",
            json={"name": "Bulk Buddy Persona One", "mode": "session_scoped"},
        )
        second = client.post(
            "/api/v1/persona/profiles",
            json={"name": "Bulk Buddy Persona Two", "mode": "session_scoped"},
        )
        assert first.status_code == 201, first.text
        assert second.status_code == 201, second.text

        first_id = first.json()["id"]
        second_id = second.json()["id"]

        real_get_persona_buddy = persona_db.get_persona_buddy
        bulk_calls: list[tuple[str, tuple[str, ...], bool]] = []

        def _bulk_lookup(*, user_id: str, persona_ids: list[str], include_deleted_personas: bool):
            bulk_calls.append((user_id, tuple(persona_ids), include_deleted_personas))
            return {
                persona_id: real_get_persona_buddy(
                    persona_id=persona_id,
                    user_id=user_id,
                    include_deleted_personas=include_deleted_personas,
                )
                for persona_id in persona_ids
            }

        monkeypatch.setattr(persona_db, "list_persona_buddies", _bulk_lookup, raising=False)
        monkeypatch.setattr(
            persona_db,
            "get_persona_buddy",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("profile list should use bulk buddy lookup")
            ),
        )

        listed = client.get("/api/v1/persona/profiles")
        assert listed.status_code == 200, listed.text
        listed_ids = {item["id"] for item in listed.json()}
        assert {first_id, second_id}.issubset(listed_ids)

    assert bulk_calls == [("1", (first_id, second_id), False)]
    fastapi_app.dependency_overrides.clear()


def test_create_persona_from_character_snapshots_origin_without_live_dependency(persona_db: CharactersRAGDB):
    character_id = persona_db.add_character_card(sample_card_data(name="Source Character"))
    assert character_id is not None

    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={
                "name": "Garden Helper",
                "character_card_id": character_id,
                "mode": "persistent_scoped",
            },
        )
        assert created.status_code == 201, created.text
        payload = created.json()
        assert payload["character_card_id"] == character_id
        assert payload["origin_character_id"] == character_id
        assert payload["origin_character_name"] == "Source Character"
        assert payload["origin_character_snapshot_at"]
        buddy_summary = payload["buddy_summary"]
        assert buddy_summary is not None
        assert buddy_summary["has_buddy"] is True
        assert buddy_summary["visual"]["species_id"]

    fastapi_app.dependency_overrides.clear()


def test_persona_session_response_includes_scope_audit(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={
                "name": "Scoped Analyst",
                "mode": "persistent_scoped",
                "system_prompt": "Use only scoped data.",
            },
        )
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]

        rules_resp = client.put(
            f"/api/v1/persona/profiles/{persona_id}/scope-rules",
            json={
                "rules": [
                    {"rule_type": "conversation_id", "rule_value": "conv-a", "include": True},
                    {"rule_type": "media_tag", "rule_value": "science", "include": True},
                    {"rule_type": "media_id", "rule_value": "999", "include": False},
                ]
            },
        )
        assert rules_resp.status_code == 200, rules_resp.text

        session = client.post("/api/v1/persona/session", json={"persona_id": persona_id})
        assert session.status_code == 200, session.text
        payload = session.json()
        assert payload["runtime_mode"] == "persistent_scoped"
        assert payload["scope_snapshot_id"]
        audit = payload.get("scope_audit") or {}
        assert audit.get("source_rule_count") == 3
        assert audit.get("include_rule_count") == 2
        assert audit.get("exclude_rule_count") == 1

        session_id = payload["session_id"]
        listed = client.get(f"/api/v1/persona/sessions?persona_id={persona_id}&limit=20")
        assert listed.status_code == 200, listed.text
        rows = listed.json()
        assert any(
            row.get("session_id") == session_id
            and row.get("runtime_mode") == "persistent_scoped"
            and row.get("scope_snapshot_id") == payload["scope_snapshot_id"]
            for row in rows
        )

        detail = client.get(f"/api/v1/persona/sessions/{session_id}")
        assert detail.status_code == 200, detail.text
        detail_payload = detail.json()
        assert detail_payload["session_id"] == session_id
        assert detail_payload["scope_snapshot_id"] == payload["scope_snapshot_id"]
        assert isinstance(detail_payload.get("scope_audit"), dict)

    fastapi_app.dependency_overrides.clear()


def test_persona_profile_state_docs_roundtrip_and_archives_previous_version(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={"name": "Persistent Companion", "mode": "persistent_scoped"},
        )
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]

        initial_state = client.get(f"/api/v1/persona/profiles/{persona_id}/state")
        assert initial_state.status_code == 200, initial_state.text
        assert initial_state.json()["soul_md"] is None
        assert initial_state.json()["identity_md"] is None
        assert initial_state.json()["heartbeat_md"] is None

        put_first = client.put(
            f"/api/v1/persona/profiles/{persona_id}/state",
            json={
                "soul_md": "You are curious and kind.",
                "identity_md": "Name: Ava",
            },
        )
        assert put_first.status_code == 200, put_first.text
        first_payload = put_first.json()
        assert first_payload["soul_md"] == "You are curious and kind."
        assert first_payload["identity_md"] == "Name: Ava"
        assert first_payload["heartbeat_md"] is None
        assert first_payload["last_modified"]

        put_second = client.put(
            f"/api/v1/persona/profiles/{persona_id}/state",
            json={"soul_md": "You are calm and reflective."},
        )
        assert put_second.status_code == 200, put_second.text
        second_payload = put_second.json()
        assert second_payload["soul_md"] == "You are calm and reflective."
        assert second_payload["identity_md"] == "Name: Ava"

        soul_active = persona_db.list_persona_memory_entries(
            user_id="1",
            persona_id=persona_id,
            memory_type="persona_state_soul",
            include_archived=False,
            include_deleted=False,
            limit=20,
            offset=0,
        )
        assert len(soul_active) == 1
        assert soul_active[0]["content"] == "You are calm and reflective."

        soul_all = persona_db.list_persona_memory_entries(
            user_id="1",
            persona_id=persona_id,
            memory_type="persona_state_soul",
            include_archived=True,
            include_deleted=False,
            limit=20,
            offset=0,
        )
        assert len(soul_all) >= 2
        assert any(
            bool(row.get("archived"))
            and str(row.get("content")) == "You are curious and kind."
            for row in soul_all
        )

    fastapi_app.dependency_overrides.clear()


def test_persona_profile_voice_defaults_roundtrip(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={
                "name": "Voice Helper",
                "mode": "persistent_scoped",
                "voice_defaults": {
                    "stt_language": "en-US",
                    "stt_model": "whisper-1",
                    "tts_provider": "tldw",
                    "tts_voice": "af_heart",
                    "confirmation_mode": "always",
                    "voice_chat_trigger_phrases": ["hey helper", "okay helper"],
                    "wake_behavior": "continuous",
                    "auto_resume": True,
                    "barge_in": False,
                    "auto_commit_enabled": True,
                    "vad_threshold": 0.35,
                    "min_silence_ms": 150,
                    "turn_stop_secs": 0.1,
                    "min_utterance_secs": 0.25,
                },
            },
        )
        assert created.status_code == 201, created.text
        payload = created.json()
        persona_id = payload["id"]
        assert payload["voice_defaults"]["stt_language"] == "en-US"
        assert payload["voice_defaults"]["confirmation_mode"] == "always"
        assert payload["voice_defaults"]["voice_chat_trigger_phrases"] == [
            "hey helper",
            "okay helper",
        ]
        assert payload["voice_defaults"]["wake_behavior"] == "continuous"
        assert payload["voice_defaults"]["auto_commit_enabled"] is True
        assert payload["voice_defaults"]["vad_threshold"] == 0.35
        assert payload["voice_defaults"]["min_silence_ms"] == 150
        assert payload["voice_defaults"]["turn_stop_secs"] == 0.1
        assert payload["voice_defaults"]["min_utterance_secs"] == 0.25

        fetched = client.get(f"/api/v1/persona/profiles/{persona_id}")
        assert fetched.status_code == 200, fetched.text
        fetched_payload = fetched.json()
        assert fetched_payload["voice_defaults"]["tts_provider"] == "tldw"
        assert fetched_payload["voice_defaults"]["tts_voice"] == "af_heart"
        assert fetched_payload["voice_defaults"]["wake_behavior"] == "continuous"
        assert fetched_payload["voice_defaults"]["auto_resume"] is True
        assert fetched_payload["voice_defaults"]["barge_in"] is False
        assert fetched_payload["voice_defaults"]["auto_commit_enabled"] is True
        assert fetched_payload["voice_defaults"]["vad_threshold"] == 0.35
        assert fetched_payload["voice_defaults"]["min_silence_ms"] == 150
        assert fetched_payload["voice_defaults"]["turn_stop_secs"] == 0.1
        assert fetched_payload["voice_defaults"]["min_utterance_secs"] == 0.25

        updated = client.patch(
            f"/api/v1/persona/profiles/{persona_id}",
            json={
                "voice_defaults": {
                    "stt_language": "fr-FR",
                    "confirmation_mode": "destructive_only",
                    "voice_chat_trigger_phrases": ["bonjour helper"],
                    "wake_behavior": "push_to_talk_after_wake",
                    "auto_resume": False,
                    "barge_in": True,
                    "auto_commit_enabled": False,
                    "vad_threshold": 0.61,
                    "min_silence_ms": 640,
                    "turn_stop_secs": 0.48,
                    "min_utterance_secs": 0.82,
                }
            },
        )
        assert updated.status_code == 200, updated.text
        updated_payload = updated.json()
        assert updated_payload["voice_defaults"]["stt_language"] == "fr-FR"
        assert updated_payload["voice_defaults"]["confirmation_mode"] == "destructive_only"
        assert updated_payload["voice_defaults"]["voice_chat_trigger_phrases"] == [
            "bonjour helper"
        ]
        assert (
            updated_payload["voice_defaults"]["wake_behavior"]
            == "push_to_talk_after_wake"
        )
        assert updated_payload["voice_defaults"]["auto_resume"] is False
        assert updated_payload["voice_defaults"]["barge_in"] is True
        assert updated_payload["voice_defaults"]["auto_commit_enabled"] is False
        assert updated_payload["voice_defaults"]["vad_threshold"] == 0.61
        assert updated_payload["voice_defaults"]["min_silence_ms"] == 640
        assert updated_payload["voice_defaults"]["turn_stop_secs"] == 0.48
        assert updated_payload["voice_defaults"]["min_utterance_secs"] == 0.82

    fastapi_app.dependency_overrides.clear()


def test_persona_profile_voice_defaults_rejects_invalid_wake_behavior(
    persona_db: CharactersRAGDB,
):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={
                "name": "Bad Wake Helper",
                "mode": "persistent_scoped",
                "voice_defaults": {"wake_behavior": "always_on_background"},
            },
        )
        assert created.status_code == 422, created.text

    fastapi_app.dependency_overrides.clear()


def test_persona_profile_voice_defaults_clamps_turn_detection_values(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={
                "name": "Voice Clamp Helper",
                "mode": "persistent_scoped",
                "voice_defaults": {
                    "auto_commit_enabled": True,
                    "vad_threshold": 8,
                    "min_silence_ms": -1,
                    "turn_stop_secs": 0.001,
                    "min_utterance_secs": -2,
                },
            },
        )
        assert created.status_code == 201, created.text
        payload = created.json()
        persona_id = payload["id"]
        assert payload["voice_defaults"]["auto_commit_enabled"] is True
        assert payload["voice_defaults"]["vad_threshold"] == 1.0
        assert payload["voice_defaults"]["min_silence_ms"] == 50
        assert payload["voice_defaults"]["turn_stop_secs"] == 0.05
        assert payload["voice_defaults"]["min_utterance_secs"] == 0.0

        updated = client.patch(
            f"/api/v1/persona/profiles/{persona_id}",
            json={
                "voice_defaults": {
                    "auto_commit_enabled": False,
                    "vad_threshold": -4,
                    "min_silence_ms": 200_000,
                    "turn_stop_secs": 99,
                    "min_utterance_secs": 11,
                }
            },
        )
        assert updated.status_code == 200, updated.text
        updated_payload = updated.json()
        assert updated_payload["voice_defaults"]["auto_commit_enabled"] is False
        assert updated_payload["voice_defaults"]["vad_threshold"] == 0.0
        assert updated_payload["voice_defaults"]["min_silence_ms"] == 10_000
        assert updated_payload["voice_defaults"]["turn_stop_secs"] == 10.0
        assert updated_payload["voice_defaults"]["min_utterance_secs"] == 10.0

    fastapi_app.dependency_overrides.clear()


def test_persona_profile_setup_run_id_defaults_and_roundtrip(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={
                "name": "Setup Wizard Persona",
                "mode": "persistent_scoped",
            },
        )
        assert created.status_code == 201, created.text
        payload = created.json()
        persona_id = payload["id"]
        assert payload["setup"] == {
            "status": "not_started",
            "version": 1,
            "run_id": None,
            "current_step": "persona",
            "completed_steps": [],
            "completed_at": None,
            "last_test_type": None,
        }

        updated = client.patch(
            f"/api/v1/persona/profiles/{persona_id}",
            json={
                "setup": {
                    "status": "in_progress",
                    "version": 1,
                    "run_id": "setup-run-1",
                    "current_step": "commands",
                    "completed_steps": ["persona", "voice"],
                    "completed_at": None,
                    "last_test_type": None,
                }
            },
        )
        assert updated.status_code == 200, updated.text
        updated_payload = updated.json()
        assert updated_payload["setup"] == {
            "status": "in_progress",
            "version": 1,
            "run_id": "setup-run-1",
            "current_step": "commands",
            "completed_steps": ["persona", "voice"],
            "completed_at": None,
            "last_test_type": None,
        }

        completed = client.patch(
            f"/api/v1/persona/profiles/{persona_id}",
            json={
                "setup": {
                    "status": "completed",
                    "version": 1,
                    "run_id": "setup-run-1",
                    "current_step": "test",
                    "completed_steps": ["persona", "voice", "commands", "safety", "test"],
                    "completed_at": "2026-03-13T10:00:00Z",
                    "last_test_type": "dry_run",
                }
            },
        )
        assert completed.status_code == 200, completed.text
        completed_payload = completed.json()
        assert completed_payload["setup"] == {
            "status": "completed",
            "version": 1,
            "run_id": "setup-run-1",
            "current_step": "test",
            "completed_steps": ["persona", "voice", "commands", "safety", "test"],
            "completed_at": "2026-03-13T10:00:00Z",
            "last_test_type": "dry_run",
        }

        fetched = client.get(f"/api/v1/persona/profiles/{persona_id}")
        assert fetched.status_code == 200, fetched.text
        fetched_payload = fetched.json()
        assert fetched_payload["setup"] == {
            "status": "completed",
            "version": 1,
            "run_id": "setup-run-1",
            "current_step": "test",
            "completed_steps": ["persona", "voice", "commands", "safety", "test"],
            "completed_at": "2026-03-13T10:00:00Z",
            "last_test_type": "dry_run",
        }

        listed = client.get("/api/v1/persona/profiles")
        assert listed.status_code == 200, listed.text
        listed_payload = listed.json()
        listed_profile = next(
            item for item in listed_payload if item["id"] == persona_id
        )
        assert listed_profile["setup"] == {
            "status": "completed",
            "version": 1,
            "run_id": "setup-run-1",
            "current_step": "test",
            "completed_steps": ["persona", "voice", "commands", "safety", "test"],
            "completed_at": "2026-03-13T10:00:00Z",
            "last_test_type": "dry_run",
        }

        reset = client.patch(
            f"/api/v1/persona/profiles/{persona_id}",
            json={
                "setup": {
                    "status": "in_progress",
                    "version": 1,
                    "run_id": "setup-run-2",
                    "current_step": "persona",
                    "completed_steps": [],
                    "completed_at": None,
                    "last_test_type": None,
                }
            },
        )
        assert reset.status_code == 200, reset.text
        reset_payload = reset.json()
        assert reset_payload["setup"] == {
            "status": "in_progress",
            "version": 1,
            "run_id": "setup-run-2",
            "current_step": "persona",
            "completed_steps": [],
            "completed_at": None,
            "last_test_type": None,
        }

        completed_live = client.patch(
            f"/api/v1/persona/profiles/{persona_id}",
            json={
                "setup": {
                    "status": "completed",
                    "version": 1,
                    "run_id": "setup-run-2",
                    "current_step": "test",
                    "completed_steps": ["persona", "voice", "commands", "safety", "test"],
                    "completed_at": "2026-03-13T10:05:00Z",
                    "last_test_type": "live_session",
                }
            },
        )
        assert completed_live.status_code == 200, completed_live.text
        completed_live_payload = completed_live.json()
        assert completed_live_payload["setup"] == {
            "status": "completed",
            "version": 1,
            "run_id": "setup-run-2",
            "current_step": "test",
            "completed_steps": ["persona", "voice", "commands", "safety", "test"],
            "completed_at": "2026-03-13T10:05:00Z",
            "last_test_type": "live_session",
        }

    fastapi_app.dependency_overrides.clear()


def test_persona_profile_state_update_rejects_empty_payload(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={"name": "State Empty Payload", "mode": "persistent_scoped"},
        )
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]

        empty_update = client.put(
            f"/api/v1/persona/profiles/{persona_id}/state",
            json={},
        )
        assert empty_update.status_code == 400

    fastapi_app.dependency_overrides.clear()


def test_persona_profile_state_update_supports_null_clear(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={"name": "State Clear Persona", "mode": "persistent_scoped"},
        )
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]

        seed_state = client.put(
            f"/api/v1/persona/profiles/{persona_id}/state",
            json={
                "soul_md": "Original soul",
                "identity_md": "Identity anchor",
            },
        )
        assert seed_state.status_code == 200, seed_state.text

        cleared = client.put(
            f"/api/v1/persona/profiles/{persona_id}/state",
            json={"soul_md": None},
        )
        assert cleared.status_code == 200, cleared.text
        payload = cleared.json()
        assert payload["soul_md"] is None
        assert payload["identity_md"] == "Identity anchor"

        active_soul = persona_db.list_persona_memory_entries(
            user_id="1",
            persona_id=persona_id,
            memory_type="persona_state_soul",
            include_archived=False,
            include_deleted=False,
            limit=20,
            offset=0,
        )
        assert active_soul == []

        archived_soul = persona_db.list_persona_memory_entries(
            user_id="1",
            persona_id=persona_id,
            memory_type="persona_state_soul",
            include_archived=True,
            include_deleted=False,
            limit=20,
            offset=0,
        )
        assert archived_soul
        assert all(bool(row.get("archived")) for row in archived_soul)

    fastapi_app.dependency_overrides.clear()


def test_persona_profile_state_history_and_restore(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={"name": "State History Persona", "mode": "persistent_scoped"},
        )
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]

        v1 = "Soul v1"
        v2 = "Soul v2"
        put_v1 = client.put(
            f"/api/v1/persona/profiles/{persona_id}/state",
            json={"soul_md": v1},
        )
        assert put_v1.status_code == 200, put_v1.text
        put_v2 = client.put(
            f"/api/v1/persona/profiles/{persona_id}/state",
            json={"soul_md": v2},
        )
        assert put_v2.status_code == 200, put_v2.text
        assert put_v2.json()["soul_md"] == v2

        history = client.get(
            f"/api/v1/persona/profiles/{persona_id}/state/history?field=soul_md&include_archived=true&limit=20"
        )
        assert history.status_code == 200, history.text
        entries = history.json()["entries"]
        assert len(entries) >= 2
        archived_v1_entries = [entry for entry in entries if not entry["is_active"] and entry["content"] == v1]
        assert archived_v1_entries
        restore_entry_id = archived_v1_entries[0]["entry_id"]

        restore = client.post(
            f"/api/v1/persona/profiles/{persona_id}/state/restore",
            json={"entry_id": restore_entry_id},
        )
        assert restore.status_code == 200, restore.text
        assert restore.json()["soul_md"] == v1

        history_active_only = client.get(
            f"/api/v1/persona/profiles/{persona_id}/state/history?field=soul_md&include_archived=false&limit=20"
        )
        assert history_active_only.status_code == 200, history_active_only.text
        active_entries = history_active_only.json()["entries"]
        assert len(active_entries) == 1
        assert active_entries[0]["is_active"] is True
        assert active_entries[0]["content"] == v1
        active_entry_id = active_entries[0]["entry_id"]

        archive = client.post(
            f"/api/v1/persona/profiles/{persona_id}/state/archive",
            json={"entry_id": active_entry_id},
        )
        assert archive.status_code == 200, archive.text
        assert archive.json()["soul_md"] is None

        history_after_archive = client.get(
            f"/api/v1/persona/profiles/{persona_id}/state/history?field=soul_md&include_archived=false&limit=20"
        )
        assert history_after_archive.status_code == 200, history_after_archive.text
        assert history_after_archive.json()["entries"] == []

    fastapi_app.dependency_overrides.clear()


def test_persona_profile_state_history_scoping_and_validation(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as user_one_client:
        created = user_one_client.post(
            "/api/v1/persona/profiles",
            json={"name": "State History Scope Persona", "mode": "persistent_scoped"},
        )
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]

        state_write = user_one_client.put(
            f"/api/v1/persona/profiles/{persona_id}/state",
            json={"identity_md": "Owner-only identity"},
        )
        assert state_write.status_code == 200, state_write.text

        invalid_filter = user_one_client.get(
            f"/api/v1/persona/profiles/{persona_id}/state/history?field=unknown_field"
        )
        assert invalid_filter.status_code == 400
        invalid_archive = user_one_client.post(
            f"/api/v1/persona/profiles/{persona_id}/state/archive",
            json={"entry_id": "   "},
        )
        assert invalid_archive.status_code == 422

    fastapi_app.dependency_overrides.clear()

    with _client_for_user(2, persona_db) as user_two_client:
        history = user_two_client.get(f"/api/v1/persona/profiles/{persona_id}/state/history")
        assert history.status_code == 404
        restore = user_two_client.post(
            f"/api/v1/persona/profiles/{persona_id}/state/restore",
            json={"entry_id": "forged-entry"},
        )
        assert restore.status_code == 404
        archive = user_two_client.post(
            f"/api/v1/persona/profiles/{persona_id}/state/archive",
            json={"entry_id": "forged-entry"},
        )
        assert archive.status_code == 404

    fastapi_app.dependency_overrides.clear()


def test_persona_profile_state_is_user_scoped(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as user_one_client:
        created = user_one_client.post(
            "/api/v1/persona/profiles",
            json={"name": "Private State Persona", "mode": "persistent_scoped"},
        )
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]
        set_state = user_one_client.put(
            f"/api/v1/persona/profiles/{persona_id}/state",
            json={"identity_md": "Visible only to owner"},
        )
        assert set_state.status_code == 200, set_state.text

    fastapi_app.dependency_overrides.clear()

    with _client_for_user(2, persona_db) as user_two_client:
        fetched = user_two_client.get(f"/api/v1/persona/profiles/{persona_id}/state")
        assert fetched.status_code == 404
        updated = user_two_client.put(
            f"/api/v1/persona/profiles/{persona_id}/state",
            json={"identity_md": "forged"},
        )
        assert updated.status_code == 404

    fastapi_app.dependency_overrides.clear()


def test_persona_profile_state_update_rejects_oversized_doc_with_413(persona_db: CharactersRAGDB, monkeypatch):
    monkeypatch.setattr(persona_ep, "_get_persona_state_doc_max_chars", lambda: 16)

    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={"name": "Oversize Guard Persona", "mode": "persistent_scoped"},
        )
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]

        oversized = client.put(
            f"/api/v1/persona/profiles/{persona_id}/state",
            json={"identity_md": "x" * 48},
        )
        assert oversized.status_code == 413
        detail = str(oversized.json().get("detail") or "")
        assert "identity_md exceeds max chars" in detail

    fastapi_app.dependency_overrides.clear()


def test_persona_profile_state_metrics_smoke(persona_db: CharactersRAGDB, monkeypatch):
    captured_metrics: list[tuple[str, dict[str, str]]] = []

    def _fake_increment(metric_name, value=1, labels=None):
        captured_metrics.append((str(metric_name), {str(k): str(v) for k, v in dict(labels or {}).items()}))
        return None

    monkeypatch.setattr(persona_ep, "increment_counter", _fake_increment)

    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={"name": "Metric Persona", "mode": "persistent_scoped"},
        )
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]

        write_resp = client.put(
            f"/api/v1/persona/profiles/{persona_id}/state",
            json={"soul_md": "metric state"},
        )
        assert write_resp.status_code == 200, write_resp.text

        read_resp = client.get(f"/api/v1/persona/profiles/{persona_id}/state")
        assert read_resp.status_code == 200, read_resp.text

    persona_state_metrics = [
        (name, labels) for name, labels in captured_metrics if name == "persona_state_docs_total"
    ]
    assert persona_state_metrics
    assert any(labels.get("action") == "write" and labels.get("result") == "success" for _, labels in persona_state_metrics)
    assert any(labels.get("action") == "read" and labels.get("result") == "success" for _, labels in persona_state_metrics)

    fastapi_app.dependency_overrides.clear()


def test_persona_profiles_are_user_scoped(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as user_one_client:
        created = user_one_client.post(
            "/api/v1/persona/profiles",
            json={"name": "Private Persona", "mode": "session_scoped"},
        )
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]

    fastapi_app.dependency_overrides.clear()

    with _client_for_user(2, persona_db) as user_two_client:
        fetched = user_two_client.get(f"/api/v1/persona/profiles/{persona_id}")
        assert fetched.status_code == 404

        sessions = user_two_client.get("/api/v1/persona/sessions")
        assert sessions.status_code == 200
        assert sessions.json() == []

    fastapi_app.dependency_overrides.clear()
