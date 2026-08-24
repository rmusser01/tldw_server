import os

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


async def _prepare_real_authnz_users(tmp_path, monkeypatch):
    """Create isolated multi-user AuthNZ credentials for a real dependency-path test."""
    db_path = tmp_path / "users.db"
    original_env = {
        name: os.environ.get(name)
        for name in ("AUTH_MODE", "DATABASE_URL", "JWT_SECRET_KEY")
    }
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("JWT_SECRET_KEY", "persona-buddy-auth-test-secret-0123456789")

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service, reset_jwt_service
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB, reset_users_db

    reset_settings()
    reset_jwt_service()
    await reset_db_pool()
    await reset_users_db()
    pool = await get_db_pool()
    ensure_authnz_tables(db_path)
    users_db = UsersDB(pool)
    await users_db.initialize()
    owner = await users_db.create_user(
        username="persona-auth-owner",
        email="persona-auth-owner@example.com",
        password_hash="not-used-by-this-test",
        is_verified=True,
    )
    other = await users_db.create_user(
        username="persona-auth-other",
        email="persona-auth-other@example.com",
        password_hash="not-used-by-this-test",
        is_verified=True,
    )
    key_manager = APIKeyManager(pool)
    owner_key = (await key_manager.create_api_key(user_id=int(owner["id"]), name="persona-owner"))["key"]
    other_key = (await key_manager.create_api_key(user_id=int(other["id"]), name="persona-other"))["key"]
    jwt_service = get_jwt_service()
    return {
        "owner": owner,
        "other": other,
        "owner_key": owner_key,
        "other_key": other_key,
        "owner_token": jwt_service.create_access_token(
            user_id=int(owner["id"]), username=str(owner["username"]), role=str(owner["role"])
        ),
        "other_token": jwt_service.create_access_token(
            user_id=int(other["id"]), username=str(other["username"]), role=str(other["role"])
        ),
        "original_env": original_env,
    }


async def _restore_real_authnz_environment(state, monkeypatch) -> None:
    """Close the isolated auth pool and invalidate setting caches after the test."""
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.jwt_service import reset_jwt_service
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import reset_users_db

    await reset_db_pool()
    await reset_users_db()
    for name, value in state["original_env"].items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    reset_settings()
    reset_jwt_service()


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
        detail = created.json()["detail"].lower()
        assert "rollback" in detail or "roll back" in detail

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
        detail = updated.json()["detail"].lower()
        assert "rollback" in detail or "roll back" in detail

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


def test_missing_global_preference_returns_expressive_default(persona_db: CharactersRAGDB):
    """A successful no-row read uses the documented calm default."""
    with _client_for_user(1, persona_db) as client:
        response = client.get("/api/v1/persona/buddy/preferences")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "ambient_mode": "expressive",
        "version": None,
        "stored": False,
    }


def test_global_preference_patch_round_trips_for_authenticated_api_key_owner(
    persona_db: CharactersRAGDB,
):
    """The authenticated owner, rather than request data, owns the preference row."""
    with _client_for_user(1, persona_db) as api_key_client:
        created = api_key_client.patch(
            "/api/v1/persona/buddy/preferences",
            json={"ambient_mode": "roaming", "expected_version": None},
        )
        assert created.status_code == 200, created.text
        assert created.json() == {"ambient_mode": "roaming", "version": 1, "stored": True}

        loaded = api_key_client.get("/api/v1/persona/buddy/preferences")

    assert loaded.status_code == 200, loaded.text
    assert loaded.json() == {"ambient_mode": "roaming", "version": 1, "stored": True}


def test_global_preference_invalid_mode_returns_validation_error(persona_db: CharactersRAGDB):
    """Only the lowercase wire-mode enum is accepted at the HTTP boundary."""
    with _client_for_user(1, persona_db) as client:
        response = client.patch(
            "/api/v1/persona/buddy/preferences",
            json={"ambient_mode": "Expressive", "expected_version": None},
        )

    assert response.status_code == 422, response.text


def test_global_preference_backend_failure_is_not_treated_as_missing_row(
    persona_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    """A broken preference read must not silently select Expressive."""
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

    def _raise_read_failure(*_args, **_kwargs):
        raise CharactersRAGDBError("preference database unavailable")

    monkeypatch.setattr(persona_db, "get_persona_buddy_preferences", _raise_read_failure)
    with _client_for_user(1, persona_db) as client:
        response = client.get("/api/v1/persona/buddy/preferences")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] != "expressive"


def test_stale_persona_override_patch_returns_conflict_for_bearer_owner(
    persona_db: CharactersRAGDB,
):
    """A versioned persona override hides stale writes and keeps current-user ownership."""
    with _client_for_user(1, persona_db) as bearer_client:
        created = bearer_client.post("/api/v1/persona/profiles", json={"name": "Override Persona"})
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]
        buddy = persona_db.get_persona_buddy(persona_id=persona_id, user_id="1")
        assert buddy is not None

        updated = bearer_client.patch(
            f"/api/v1/persona/profiles/{persona_id}/buddy/preferences",
            json={"ambient_mode": "roaming", "expected_version": buddy["version"]},
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["ambient_mode"] == "roaming"

        stale = bearer_client.patch(
            f"/api/v1/persona/profiles/{persona_id}/buddy/preferences",
            json={"ambient_mode": "off", "expected_version": buddy["version"]},
        )

    assert stale.status_code == 409, stale.text


def test_persona_override_get_and_clear_preserve_unknown_overlay_keys(
    persona_db: CharactersRAGDB,
):
    """Use-global reads and clears only ambient_mode on the owned Buddy row."""
    with _client_for_user(1, persona_db) as client:
        created = client.post("/api/v1/persona/profiles", json={"name": "Global Override Persona"})
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]
        buddy = persona_db.get_persona_buddy(persona_id=persona_id, user_id="1")
        assert buddy is not None

        missing = client.get(
            f"/api/v1/persona/profiles/{persona_id}/buddy/preferences"
        )
        assert missing.status_code == 200, missing.text
        assert missing.json() == {
            "ambient_mode": None,
            "version": buddy["version"],
            "stored": False,
        }

        seeded = persona_db.patch_persona_buddy_overlay_preferences(
            persona_id=persona_id,
            user_id="1",
            patch={
                "ambient_mode": "roaming",
                "accessory_id": "scarf",
                "future_overlay": {"kept": True},
            },
            expected_version=buddy["version"],
        )
        loaded = client.get(
            f"/api/v1/persona/profiles/{persona_id}/buddy/preferences"
        )
        assert loaded.status_code == 200, loaded.text
        assert loaded.json() == {
            "ambient_mode": "roaming",
            "version": seeded["version"],
            "stored": True,
        }

        cleared = client.patch(
            f"/api/v1/persona/profiles/{persona_id}/buddy/preferences",
            json={"ambient_mode": None, "expected_version": seeded["version"]},
        )

    assert cleared.status_code == 200, cleared.text
    assert cleared.json() == {
        "ambient_mode": None,
        "version": seeded["version"] + 1,
        "stored": False,
    }
    persisted = persona_db.get_persona_buddy(persona_id=persona_id, user_id="1")
    assert persisted is not None
    assert "ambient_mode" not in persisted["overlay_preferences"]
    assert persisted["overlay_preferences"]["accessory_id"] == "scarf"
    assert persisted["overlay_preferences"]["future_overlay"] == {"kept": True}


def test_persona_override_clear_rejects_stale_buddy_version(
    persona_db: CharactersRAGDB,
):
    """Clearing to Use global participates in the same optimistic version contract."""
    with _client_for_user(1, persona_db) as client:
        created = client.post("/api/v1/persona/profiles", json={"name": "Stale Clear Persona"})
        assert created.status_code == 201, created.text
        persona_id = created.json()["id"]
        buddy = persona_db.get_persona_buddy(persona_id=persona_id, user_id="1")
        assert buddy is not None
        updated = client.patch(
            f"/api/v1/persona/profiles/{persona_id}/buddy/preferences",
            json={"ambient_mode": "off", "expected_version": buddy["version"]},
        )
        assert updated.status_code == 200, updated.text

        stale = client.patch(
            f"/api/v1/persona/profiles/{persona_id}/buddy/preferences",
            json={"ambient_mode": None, "expected_version": buddy["version"]},
        )

    assert stale.status_code == 409, stale.text


@pytest.mark.parametrize("invalid_version", [True, 1.0, "1"])
def test_preference_expected_versions_reject_non_integer_json_values(
    persona_db: CharactersRAGDB,
    invalid_version: object,
):
    """Version fields must not coerce booleans, floats, or strings at the route boundary."""
    with _client_for_user(1, persona_db) as client:
        response = client.patch(
            "/api/v1/persona/buddy/preferences",
            json={"ambient_mode": "roaming", "expected_version": invalid_version},
        )
        assert response.status_code == 422, response.text

        created = client.post("/api/v1/persona/profiles", json={"name": "Strict Override Persona"})
        assert created.status_code == 201, created.text
        override = client.patch(
            f"/api/v1/persona/profiles/{created.json()['id']}/buddy/preferences",
            json={"ambient_mode": "roaming", "expected_version": invalid_version},
        )

    assert override.status_code == 422, override.text


@pytest.mark.asyncio
async def test_real_api_key_and_bearer_authenticate_owner_and_hide_foreign_persona(
    persona_db: CharactersRAGDB,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Exercise the real AuthNZ dependency for both supported credential headers."""
    state = await _prepare_real_authnz_users(tmp_path, monkeypatch)
    owner_id = str(state["owner"]["id"])
    persona_id = persona_db.create_persona_profile({"user_id": owner_id, "name": "Real Auth Persona"})
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: persona_db
    try:
        with TestClient(fastapi_app) as client:
            api_key_owner = client.get(
                f"/api/v1/persona/profiles/{persona_id}/buddy",
                headers={"X-API-KEY": state["owner_key"]},
            )
            api_key_other = client.get(
                f"/api/v1/persona/profiles/{persona_id}/buddy",
                headers={"X-API-KEY": state["other_key"]},
            )
            bearer_owner = client.get(
                f"/api/v1/persona/profiles/{persona_id}/buddy",
                headers={"Authorization": f"Bearer {state['owner_token']}"},
            )
            bearer_other = client.get(
                f"/api/v1/persona/profiles/{persona_id}/buddy",
                headers={"Authorization": f"Bearer {state['other_token']}"},
            )
    finally:
        fastapi_app.dependency_overrides.clear()
        await _restore_real_authnz_environment(state, monkeypatch)

    assert api_key_owner.status_code == 200, api_key_owner.text
    assert api_key_other.status_code == 404, api_key_other.text
    assert bearer_owner.status_code == 200, bearer_owner.text
    assert bearer_other.status_code == 404, bearer_other.text
