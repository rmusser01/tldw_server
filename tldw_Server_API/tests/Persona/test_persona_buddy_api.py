import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit

fastapi_app = FastAPI()
fastapi_app.include_router(persona_ep.router, prefix="/api/v1/persona")


def _client_for_user(user_id: int, db: CharactersRAGDB) -> TestClient:
    async def override_user():
        return User(id=user_id, username=f"persona-buddy-user-{user_id}", email=None, is_active=True)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    return TestClient(fastapi_app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    fastapi_app.dependency_overrides.clear()

@pytest.fixture()
def persona_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "persona_buddy_api.db"), client_id="persona-buddy-api-tests")
    yield db
    db.close_connection()


def test_get_buddy_lazily_creates_for_preexisting_persona_without_row(persona_db: CharactersRAGDB):
    persona_id = persona_db.create_persona_profile({"user_id": "1", "name": "Lazy Buddy Persona"})
    assert persona_db.get_persona_buddy(persona_id=persona_id, user_id="1") is None

    with _client_for_user(1, persona_db) as client:
        before_profile = client.get(f"/api/v1/persona/profiles/{persona_id}")
        assert before_profile.status_code == 200, before_profile.text
        before_payload = before_profile.json()
        before_buddy_summary = before_payload.get("buddy_summary")
        if before_buddy_summary is not None:
            assert before_buddy_summary["has_buddy"] is False
            assert before_buddy_summary["persona_name"] == "Lazy Buddy Persona"

        buddy_response = client.get(f"/api/v1/persona/profiles/{persona_id}/buddy")
        assert buddy_response.status_code == 200, buddy_response.text
        payload = buddy_response.json()
        assert payload["persona_id"] == persona_id
        assert "resolved_profile" in payload

        after_profile = client.get(f"/api/v1/persona/profiles/{persona_id}")
        assert after_profile.status_code == 200, after_profile.text
        after_payload = after_profile.json()
        assert after_payload["version"] == before_payload["version"]
        after_buddy_summary = after_payload["buddy_summary"]
        assert after_buddy_summary is not None
        assert after_buddy_summary["has_buddy"] is True
        assert after_buddy_summary["persona_name"] == "Lazy Buddy Persona"
        assert after_buddy_summary["role_summary"]
        assert after_buddy_summary["visual"]["species_id"]

    persisted = persona_db.get_persona_buddy(persona_id=persona_id, user_id="1")
    assert persisted is not None
    assert persisted["resolved_profile"] == payload["resolved_profile"]

def test_api_create_keeps_buddy_row_aligned_immediately(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={"name": "Create Buddy API Persona", "mode": "session_scoped"},
        )
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]

    buddy_row = persona_db.get_persona_buddy(persona_id=persona_id, user_id="1")
    assert buddy_row is not None

def test_api_create_rolls_back_visible_profile_when_buddy_upkeep_fails(persona_db: CharactersRAGDB, monkeypatch):
    def _raise_buddy_failure(*_args, **_kwargs):
        raise ValueError("buddy unavailable")

    monkeypatch.setattr(persona_ep, "ensure_persona_buddy_for_profile", _raise_buddy_failure)

    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={"name": "Create Best Effort Persona", "mode": "session_scoped"},
        )
        assert created.status_code == 400, created.text
        assert created.json()["detail"] == "Persona buddy validation failed"

    active_profiles = persona_db.list_persona_profiles(user_id="1", include_deleted=False, limit=20, offset=0)
    assert not any(profile["name"] == "Create Best Effort Persona" for profile in active_profiles)

    deleted_profiles = persona_db.list_persona_profiles(user_id="1", include_deleted=True, limit=20, offset=0)
    rolled_back = next(profile for profile in deleted_profiles if profile["name"] == "Create Best Effort Persona")
    assert rolled_back["deleted"] is True
    assert persona_db.get_persona_buddy(persona_id=rolled_back["id"], user_id="1") is None

def test_api_update_keeps_buddy_row_aligned_after_stable_input_change(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={"name": "Update Buddy API Persona"},
        )
        assert created.status_code == 201, created.text
        created_payload = created.json()
        persona_id = created_payload["id"]

        before = persona_db.get_persona_buddy(persona_id=persona_id, user_id="1")
        assert before is not None

        updated = client.patch(
            f"/api/v1/persona/profiles/{persona_id}",
            params={"expected_version": int(created_payload["version"])},
            json={"name": "Update Buddy API Persona Renamed"},
        )
        assert updated.status_code == 200, updated.text

    after = persona_db.get_persona_buddy(persona_id=persona_id, user_id="1")
    assert after is not None
    assert after["source_fingerprint"] != before["source_fingerprint"]
    assert int(after["version"]) > int(before["version"])

def test_system_prompt_only_updates_do_not_rederive_buddy(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={"name": "Prompt Stable Buddy Persona"},
        )
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]

        buddy_before = client.get(f"/api/v1/persona/profiles/{persona_id}/buddy")
        assert buddy_before.status_code == 200, buddy_before.text
        before_payload = buddy_before.json()

        updated = client.patch(
            f"/api/v1/persona/profiles/{persona_id}",
            params={"expected_version": int(created.json()["version"])},
            json={"system_prompt": "This prompt changed, but buddy identity should not."},
        )
        assert updated.status_code == 200, updated.text

        buddy_after = client.get(f"/api/v1/persona/profiles/{persona_id}/buddy")
        assert buddy_after.status_code == 200, buddy_after.text
        after_payload = buddy_after.json()

    assert after_payload["resolved_profile"] == before_payload["resolved_profile"]
    assert after_payload["last_modified"] == before_payload["last_modified"]

def test_api_update_reverts_profile_when_buddy_upkeep_fails(persona_db: CharactersRAGDB, monkeypatch):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={"name": "Update Best Effort Persona"},
        )
        assert created.status_code == 201, created.text
        created_payload = created.json()
        persona_id = created_payload["id"]

    profile = persona_db.get_persona_profile(persona_id, user_id="1")
    buddy_before = persona_db.get_persona_buddy(persona_id=persona_id, user_id="1")
    assert profile is not None
    assert buddy_before is not None

    def _raise_buddy_failure(*_args, **_kwargs):
        raise ValueError("buddy unavailable")

    monkeypatch.setattr(persona_ep, "ensure_persona_buddy_for_profile", _raise_buddy_failure)

    with _client_for_user(1, persona_db) as client:
        updated = client.patch(
            f"/api/v1/persona/profiles/{persona_id}",
            params={"expected_version": int(created_payload["version"])},
            json={"name": "Update Best Effort Persona Renamed"},
        )
        assert updated.status_code == 400, updated.text
        assert updated.json()["detail"] == "Persona buddy validation failed"

    refreshed = persona_db.get_persona_profile(persona_id, user_id="1")
    buddy_after = persona_db.get_persona_buddy(persona_id=persona_id, user_id="1")
    assert refreshed is not None
    assert refreshed["name"] == "Update Best Effort Persona"
    assert buddy_after is not None
    assert buddy_after["source_fingerprint"] == buddy_before["source_fingerprint"]

def test_api_create_surfaces_rollback_failures_after_buddy_validation_error(
    persona_db: CharactersRAGDB,
    monkeypatch,
):
    def _raise_buddy_failure(*_args, **_kwargs):
        raise ValueError("buddy unavailable")

    monkeypatch.setattr(persona_ep, "ensure_persona_buddy_for_profile", _raise_buddy_failure)
    monkeypatch.setattr(
        persona_db,
        "soft_delete_persona_profile",
        lambda **_kwargs: False,
    )

    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={"name": "Create Rollback Failure Persona", "mode": "session_scoped"},
        )
        assert created.status_code == 500, created.text
        assert "rollback" in created.json()["detail"].lower()

    fastapi_app.dependency_overrides.clear()


def test_api_update_surfaces_rollback_failures_after_buddy_validation_error(
    persona_db: CharactersRAGDB,
    monkeypatch,
):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={"name": "Update Rollback Failure Persona"},
        )
        assert created.status_code == 201, created.text
        created_payload = created.json()
        persona_id = created_payload["id"]

    def _raise_buddy_failure(*_args, **_kwargs):
        raise ValueError("buddy unavailable")

    real_update = persona_db.update_persona_profile
    update_calls = {"count": 0}

    def _fail_second_update(*args, **kwargs):
        update_calls["count"] += 1
        if update_calls["count"] == 1:
            return real_update(*args, **kwargs)
        return False

    monkeypatch.setattr(persona_ep, "ensure_persona_buddy_for_profile", _raise_buddy_failure)
    monkeypatch.setattr(persona_db, "update_persona_profile", _fail_second_update)

    with _client_for_user(1, persona_db) as client:
        updated = client.patch(
            f"/api/v1/persona/profiles/{persona_id}",
            params={"expected_version": int(created_payload["version"])},
            json={"name": "Update Rollback Failure Persona Renamed"},
        )
        assert updated.status_code == 500, updated.text
        assert "rollback" in updated.json()["detail"].lower()

    fastapi_app.dependency_overrides.clear()


def test_deleted_persona_hides_buddy_until_restore_and_restore_preserves_buddy_response(
    persona_db: CharactersRAGDB,
):
    with _client_for_user(1, persona_db) as client:
        created = client.post("/api/v1/persona/profiles", json={"name": "Delete Restore Buddy API Persona"})
        assert created.status_code == 201, created.text
        created_payload = created.json()
        persona_id = created_payload["id"]

        before = client.get(f"/api/v1/persona/profiles/{persona_id}/buddy")
        assert before.status_code == 200, before.text
        before_payload = before.json()
        buddy_row_before_delete = persona_db.get_persona_buddy(persona_id=persona_id, user_id="1")
        assert buddy_row_before_delete is not None

        deleted = client.delete(
            f"/api/v1/persona/profiles/{persona_id}",
            params={"expected_version": int(created_payload["version"])},
        )
        assert deleted.status_code == 200, deleted.text

        hidden = client.get(f"/api/v1/persona/profiles/{persona_id}/buddy")
        assert hidden.status_code == 404, hidden.text

        deleted_profile = persona_db.get_persona_profile(persona_id, user_id="1", include_deleted=True)
        assert deleted_profile is not None
        restored = client.post(
            f"/api/v1/persona/profiles/{persona_id}/restore",
            params={"expected_version": int(deleted_profile["version"])},
        )
        assert restored.status_code == 200, restored.text
        restored_payload = restored.json()
        assert restored_payload["is_active"] is True
        restored_buddy_summary = restored_payload["buddy_summary"]
        assert restored_buddy_summary is not None
        assert restored_buddy_summary["has_buddy"] is True
        assert restored_buddy_summary["persona_name"] == "Delete Restore Buddy API Persona"
        assert restored_buddy_summary["role_summary"]
        assert restored_buddy_summary["visual"]["species_id"]

        after = client.get(f"/api/v1/persona/profiles/{persona_id}/buddy")
        assert after.status_code == 200, after.text
        after_payload = after.json()

    buddy_row_after_restore = persona_db.get_persona_buddy(persona_id=persona_id, user_id="1")
    assert buddy_row_after_restore is not None
    assert int(buddy_row_after_restore["version"]) == int(buddy_row_before_delete["version"])
    assert before_payload == after_payload


def test_restore_does_not_invoke_buddy_realignment(persona_db: CharactersRAGDB, monkeypatch):
    with _client_for_user(1, persona_db) as client:
        created = client.post("/api/v1/persona/profiles", json={"name": "Restore No Buddy Sync Persona"})
        assert created.status_code == 201, created.text
        created_payload = created.json()
        persona_id = created_payload["id"]

        deleted = client.delete(
            f"/api/v1/persona/profiles/{persona_id}",
            params={"expected_version": int(created_payload["version"])},
        )
        assert deleted.status_code == 200, deleted.text

    deleted_profile = persona_db.get_persona_profile(persona_id, user_id="1", include_deleted=True)
    assert deleted_profile is not None

    def _raise_unexpected_restore_sync(*_args, **_kwargs):
        raise AssertionError("restore should not invoke buddy realignment")

    monkeypatch.setattr(persona_ep, "_ensure_persona_buddy_after_profile_mutation", _raise_unexpected_restore_sync)

    with _client_for_user(1, persona_db) as client:
        restored = client.post(
            f"/api/v1/persona/profiles/{persona_id}/restore",
            params={"expected_version": int(deleted_profile["version"])},
        )
        assert restored.status_code == 200, restored.text
        assert restored.json()["is_active"] is True


def test_non_owner_access_to_buddy_and_restore_returns_404(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as owner_client:
        created = owner_client.post("/api/v1/persona/profiles", json={"name": "Owner Persona"})
        assert created.status_code == 201, created.text
        owner_payload = created.json()
        persona_id = owner_payload["id"]

        owner_buddy = owner_client.get(f"/api/v1/persona/profiles/{persona_id}/buddy")
        assert owner_buddy.status_code == 200, owner_buddy.text

        deleted = owner_client.delete(
            f"/api/v1/persona/profiles/{persona_id}",
            params={"expected_version": int(owner_payload["version"])},
        )
        assert deleted.status_code == 200, deleted.text

    with _client_for_user(2, persona_db) as other_client:
        hidden_buddy = other_client.get(f"/api/v1/persona/profiles/{persona_id}/buddy")
        assert hidden_buddy.status_code == 404, hidden_buddy.text

        denied_restore = other_client.post(
            f"/api/v1/persona/profiles/{persona_id}/restore",
            params={"expected_version": 1},
        )
        assert denied_restore.status_code == 404, denied_restore.text
