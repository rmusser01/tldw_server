from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints.vn_policy import router as vn_policy_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture
def chacha_dbs(tmp_path) -> Iterator[dict[int, CharactersRAGDB]]:
    databases = {
        42: CharactersRAGDB(str(tmp_path / "user-42" / "ChaChaNotes.db"), client_id="vn-policy-user-42"),
        7: CharactersRAGDB(str(tmp_path / "user-7" / "ChaChaNotes.db"), client_id="vn-policy-user-7"),
    }
    yield databases
    for database in databases.values():
        database.close_connection()


@pytest.fixture
def current_user() -> dict[str, Any]:
    return {
        "id": 42,
        "username": "user-42",
        "role": "user",
        "roles": ["user"],
        "permissions": [],
        "is_admin": False,
    }


@pytest.fixture
def authnz_pool(tmp_path) -> Iterator[DatabasePool]:
    pool = DatabasePool(Settings(AUTH_MODE="single_user", DATABASE_URL=f"sqlite:///{tmp_path / 'authnz.db'}"))
    yield pool
    asyncio.run(pool.close())


@pytest.fixture
def client(
    chacha_dbs: dict[int, CharactersRAGDB],
    current_user: dict[str, Any],
    authnz_pool: DatabasePool,
) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(vn_policy_router, prefix="/api/v1/vn")

    async def override_user() -> User:
        return User(**current_user)

    async def override_chacha_db() -> CharactersRAGDB:
        return chacha_dbs[int(current_user["id"])]

    async def override_db_pool() -> DatabasePool:
        return authnz_pool

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db
    app.dependency_overrides[get_db_pool] = override_db_pool

    with TestClient(app) as test_client:
        yield test_client


def test_list_policy_profiles_returns_offset_pagination(client: TestClient) -> None:
    response = client.get("/api/v1/vn/vn-policy/profiles?limit=1&offset=0")

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1
    assert body["limit"] == 1
    assert body["offset"] == 0
    assert body["has_more"] is True
    assert body["next_offset"] == 1
    assert body["pagination"]["mode"] == "offset"
    assert body["pagination"]["limit"] == 1


def test_read_only_profile_list_does_not_create_user_policy_definition_tables(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    response = client.get("/api/v1/vn/vn-policy/profiles?limit=20")

    cursor = chacha_dbs[42].execute_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('vn_policy_profiles', 'vn_generation_profiles')"
    )

    assert response.status_code == 200
    assert cursor.fetchall() == []


def test_normal_user_cannot_create_policy_profile(client: TestClient) -> None:
    response = client.post(
        "/api/v1/vn/vn-policy/profiles",
        json={
            "profile_id": "normal_user_profile",
            "display_name": "Normal User Profile",
            "definition": {
                "character_safety": {
                    "missing": {"general": "warn", "mature": "block"},
                    "unknown_or_ambiguous": {"general": "warn", "mature": "block"},
                    "conflicting": {"default": "block"},
                    "imported_untrusted": {"general": "warn", "mature": "block"},
                }
            },
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "permission_denied"


def test_normal_user_cannot_mutate_generation_profile(client: TestClient) -> None:
    create_response = client.post(
        "/api/v1/vn/vn-policy/generation-profiles",
        json={
            "profile_id": "normal_story",
            "display_name": "Normal Story",
            "provider": "local",
            "model": "gemma-3-12b",
            "supports_structured_output": True,
            "temperature_default": 0.7,
            "temperature_min": 0,
            "temperature_max": 1,
            "max_output_tokens": 1024,
            "allowed_content_ratings": ["general", "teen"],
            "max_choices": 4,
            "max_branch_depth": 8,
            "max_model_expansion_scope": "scene",
            "tts_allowed": True,
            "output_persistence_max_days": 30,
            "audit_mode": "metadata",
        },
    )
    patch_response = client.patch(
        "/api/v1/vn/vn-policy/generation-profiles/story_default",
        json={"display_name": "Nope"},
    )
    delete_response = client.delete("/api/v1/vn/vn-policy/generation-profiles/story_default")

    assert create_response.status_code == 403
    assert patch_response.status_code == 403
    assert delete_response.status_code == 403


def test_admin_created_policy_profile_is_visible_to_other_users(
    client: TestClient,
    current_user: dict[str, Any],
) -> None:
    current_user.update(
        {
            "id": 7,
            "username": "admin-7",
            "role": "admin",
            "roles": ["admin"],
            "permissions": ["system.configure"],
            "is_admin": True,
        }
    )
    create_response = client.post(
        "/api/v1/vn/vn-policy/profiles",
        json={
            "profile_id": "global_warn",
            "display_name": "Global Warn",
            "definition": {
                "character_safety": {
                    "missing": {"general": "warn", "mature": "block"},
                    "unknown_or_ambiguous": {"general": "warn", "mature": "block"},
                    "conflicting": {"default": "block"},
                    "imported_untrusted": {"general": "warn", "mature": "block"},
                }
            },
        },
    )

    current_user.update(
        {
            "id": 42,
            "username": "user-42",
            "role": "user",
            "roles": ["user"],
            "permissions": [],
            "is_admin": False,
        }
    )
    list_response = client.get("/api/v1/vn/vn-policy/profiles?limit=20")

    assert create_response.status_code == 201
    assert "global_warn" in {item["profile_id"] for item in list_response.json()["items"]}


def test_admin_can_create_patch_and_disable_policy_profile(
    client: TestClient,
    current_user: dict[str, Any],
) -> None:
    current_user.update(
        {
            "role": "admin",
            "roles": ["admin"],
            "permissions": ["system.configure"],
            "is_admin": True,
        }
    )

    create_response = client.post(
        "/api/v1/vn/vn-policy/profiles",
        json={
            "profile_id": "admin_policy",
            "display_name": "Admin Policy",
            "definition": {
                "character_safety": {
                    "missing": {"general": "warn", "mature": "block"},
                    "unknown_or_ambiguous": {"general": "warn", "mature": "block"},
                    "conflicting": {"default": "block"},
                    "imported_untrusted": {"general": "warn", "mature": "block"},
                }
            },
        },
    )
    patch_response = client.patch(
        "/api/v1/vn/vn-policy/profiles/admin_policy",
        json={"display_name": "Admin Policy Updated"},
    )
    delete_response = client.delete("/api/v1/vn/vn-policy/profiles/admin_policy")
    list_response = client.get("/api/v1/vn/vn-policy/profiles?limit=20")

    assert create_response.status_code == 201
    assert patch_response.status_code == 200
    assert patch_response.json()["display_name"] == "Admin Policy Updated"
    assert patch_response.json()["version"] == 2
    assert delete_response.status_code == 204
    assert "admin_policy" not in {item["profile_id"] for item in list_response.json()["items"]}


def test_admin_can_create_patch_and_disable_generation_profile(
    client: TestClient,
    current_user: dict[str, Any],
) -> None:
    current_user.update(
        {
            "role": "admin",
            "roles": ["admin"],
            "permissions": ["system.configure"],
            "is_admin": True,
        }
    )

    create_response = client.post(
        "/api/v1/vn/vn-policy/generation-profiles",
        json={
            "profile_id": "admin_story",
            "display_name": "Admin Story",
            "provider": "local",
            "model": "gemma-3-12b",
            "supports_structured_output": True,
            "temperature_default": 0.7,
            "temperature_min": 0,
            "temperature_max": 1,
            "max_output_tokens": 1024,
            "allowed_content_ratings": ["general", "teen"],
            "max_choices": 4,
            "max_branch_depth": 8,
            "max_model_expansion_scope": "scene",
            "tts_allowed": True,
            "output_persistence_max_days": 30,
            "audit_mode": "metadata",
        },
    )
    patch_response = client.patch(
        "/api/v1/vn/vn-policy/generation-profiles/admin_story",
        json={"display_name": "Admin Story Updated"},
    )
    delete_response = client.delete("/api/v1/vn/vn-policy/generation-profiles/admin_story")
    list_response = client.get("/api/v1/vn/vn-policy/generation-profiles?limit=20")

    assert create_response.status_code == 201
    assert create_response.json()["profile_id"] == "admin_story"
    assert patch_response.status_code == 200
    assert patch_response.json()["display_name"] == "Admin Story Updated"
    assert patch_response.json()["version"] == 2
    assert delete_response.status_code == 204
    assert "admin_story" not in {item["profile_id"] for item in list_response.json()["items"]}


def test_admin_invalid_policy_profile_returns_stable_vn_error(
    client: TestClient,
    current_user: dict[str, Any],
) -> None:
    current_user.update(
        {
            "role": "admin",
            "roles": ["admin"],
            "permissions": ["system.configure"],
            "is_admin": True,
        }
    )

    response = client.post(
        "/api/v1/vn/vn-policy/profiles",
        json={
            "profile_id": "bad_policy",
            "display_name": "Bad Policy",
            "definition": {"not_character_safety": {}},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_request"
    assert response.json()["detail"]["details"]["reason"] == "invalid_policy_profile"


def test_evaluate_warns_for_missing_general_metadata(client: TestClient) -> None:
    response = client.post(
        "/api/v1/vn/vn-policy/evaluate",
        json={
            "target_type": "session_setup",
            "policy_profile_id": "local_default",
            "context": {
                "content_rating": "general",
                "character_safety": {"metadata_status": "missing"},
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "warn"
    assert body["blocked"] is False
    assert body["requires_acknowledgement"] is True
    assert body["reasons"][0]["code"] == "character_safety_missing"


def test_evaluate_treats_omitted_character_safety_as_missing(client: TestClient) -> None:
    response = client.post(
        "/api/v1/vn/vn-policy/evaluate",
        json={
            "target_type": "session_setup",
            "policy_profile_id": "local_default",
            "context": {"content_rating": "general"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "warn"
    assert body["reasons"][0]["code"] == "character_safety_missing"


def test_evaluate_strict_hosted_blocks_omitted_character_safety(client: TestClient) -> None:
    response = client.post(
        "/api/v1/vn/vn-policy/evaluate",
        json={
            "target_type": "session_setup",
            "policy_profile_id": "strict_hosted",
            "context": {"content_rating": "general"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "block"
    assert body["reasons"][0]["code"] == "character_safety_missing"


def test_evaluate_rejects_cross_owner_target_context(client: TestClient) -> None:
    response = client.post(
        "/api/v1/vn/vn-policy/evaluate",
        json={
            "target_type": "script_draft",
            "target_id": 17,
            "policy_profile_id": "local_default",
            "context": {
                "target_owner_user_id": 7,
                "content_rating": "general",
                "character_safety": {"metadata_status": "adult"},
            },
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "not_found"


def test_evaluate_rejects_target_ids_without_authoritative_resolver(client: TestClient) -> None:
    response = client.post(
        "/api/v1/vn/vn-policy/evaluate",
        json={
            "target_type": "script_draft",
            "target_id": 17,
            "policy_profile_id": "local_default",
            "context": {
                "content_rating": "general",
                "character_safety": {"metadata_status": "adult"},
            },
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_request"
    assert response.json()["detail"]["details"]["reason"] == "target_resolution_unavailable"
