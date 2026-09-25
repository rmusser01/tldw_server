"""Synthetic email uploads through the production router and real AuthNZ dependencies."""

from secrets import token_urlsafe

import pytest

from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import AuthnzStorageQuotasRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.tests.MediaIngestion_NEW.integration.test_email_offline_ingestion import (
    OFFLINE_OPTIONS,
    synthetic_message,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
pytest_plugins = ["tldw_Server_API.tests.MediaIngestion_NEW.integration.test_email_authenticated_access"]


async def _upload(env, key: str | None, number: int, *, headers: dict[str, str] | None = None):
    request_headers = dict(headers or {})
    if key is not None:
        request_headers["X-API-KEY"] = key
    return await env.client.post(
        "/api/v1/media/add",
        headers=request_headers,
        files={"files": (f"synthetic-{number}.eml", synthetic_message(number).as_bytes(), "message/rfc822")},
        data=OFFLINE_OPTIONS,
    )


async def _count(env, user):
    response = await env.client.get("/api/v1/email/search", headers={"X-API-KEY": user.key["key"]})
    assert response.status_code == 200, response.text
    return response.json()["pagination"]["total"]


async def test_write_key_upload_is_persisted_only_for_its_owner(authenticated_email):
    env = authenticated_email
    alice, bob = env.users
    write_key = await env.manager.create_api_key(user_id=alice.id, name="synthetic-write", scope="write")
    response = await _upload(env, write_key["key"], 9001)
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["status"] == "Success", result
    assert result["db_id"] is not None
    assert await _count(env, alice) == len(alice.message_ids) + 1
    assert await _count(env, bob) == len(bob.message_ids)
    foreign = await env.client.get(f"/api/v1/email/messages/{result['db_id']}", headers={"X-API-KEY": bob.key["key"]})
    assert foreign.status_code == 404, foreign.text


@pytest.mark.parametrize("key", [None, "invalid", "read"])
async def test_upload_rejects_missing_invalid_and_read_only_keys(authenticated_email, key):
    env = authenticated_email
    alice = env.users[0]
    before = await _count(env, alice)
    credential = alice.key["key"] if key == "read" else token_urlsafe(32) if key == "invalid" else None
    response = await _upload(env, credential, 9002)
    assert response.status_code == (403 if key == "read" else 401), response.text
    assert await _count(env, alice) == before


async def test_viewer_role_cannot_upload_even_with_write_key(authenticated_email):
    env = authenticated_email
    viewer_id = await AuthnzUsersRepo(env.pool).create_user(
        username="synthetic_viewer",
        email="viewer@example.test",
        password_hash=token_urlsafe(32),
        role="viewer",
        is_verified=True,
    )
    key = await env.manager.create_api_key(user_id=viewer_id, name="viewer-write", scope="write")
    response = await _upload(env, key["key"], 9003)
    assert response.status_code == 403, response.text


async def test_expected_user_mismatch_rejects_before_persistence(authenticated_email):
    env = authenticated_email
    alice, bob = env.users
    write_key = await env.manager.create_api_key(user_id=alice.id, name="synthetic-write", scope="write")
    before = await _count(env, alice)
    response = await _upload(env, write_key["key"], 9004, headers={"X-TLDW-Expected-User-ID": str(bob.id)})
    assert response.status_code == 412, response.text
    assert await _count(env, alice) == before


async def test_exhausted_org_storage_quota_rejects_before_persistence(authenticated_email):
    env = authenticated_email
    alice = env.users[0]
    org_repo = AuthnzOrgsTeamsRepo(env.pool)
    org = await org_repo.create_organization(
        name="Synthetic Upload Quota", owner_user_id=alice.id, slug="synthetic-upload-quota"
    )
    await org_repo.add_org_member(org_id=org["id"], user_id=alice.id, role="owner")
    await AuthnzStorageQuotasRepo(env.pool).upsert_org_quota(org["id"], quota_mb=0)
    write_key = await env.manager.create_api_key(user_id=alice.id, name="quota-write", scope="write")
    before = await _count(env, alice)
    response = await _upload(env, write_key["key"], 9005)
    assert response.status_code == 413, response.text
    assert response.json()["detail"]["error"] == "storage_quota_exceeded"
    assert await _count(env, alice) == before


async def test_org_scoped_upload_is_searchable_with_same_credentials(authenticated_email):
    env = authenticated_email
    alice = env.users[0]
    org_repo = AuthnzOrgsTeamsRepo(env.pool)
    org = await org_repo.create_organization(
        name="Synthetic Allowed Upload", owner_user_id=alice.id, slug="synthetic-allowed-upload"
    )
    await org_repo.add_org_member(org_id=org["id"], user_id=alice.id, role="owner")
    quota_repo = AuthnzStorageQuotasRepo(env.pool)
    await quota_repo.upsert_org_quota(org["id"], quota_mb=1024)
    await quota_repo.update_org_used_mb(org["id"], 900)
    write_key = await env.manager.create_api_key(user_id=alice.id, name="org-write", scope="write")
    response = await _upload(env, write_key["key"], 9006)
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["status"] == "Success"
    assert "soft limit" in response.headers.get("X-Storage-Warning", "").lower()
    assert "X-Billing-Limit" in response.headers
    search = await env.client.get(
        "/api/v1/email/search",
        headers={"X-API-KEY": alice.key["key"]},
        params={"q": "subject:9006"},
    )
    assert search.status_code == 200, search.text
    assert [row["subject"] for row in search.json()["items"]] == ["Synthetic café report 9006"]


@pytest.mark.parametrize("authenticated_email", ["main"], indirect=True)
async def test_main_app_upload_route_rejects_read_key(authenticated_email):
    env = authenticated_email
    alice = env.users[0]
    response = await _upload(env, alice.key["key"], 9007)
    assert response.status_code == 403, response.text
    assert await _count(env, alice) == len(alice.message_ids)


async def test_same_message_in_two_orgs_uses_selected_org_for_quota_and_search(authenticated_email):
    env = authenticated_email
    alice = env.users[0]
    org_repo = AuthnzOrgsTeamsRepo(env.pool)
    quota_repo = AuthnzStorageQuotasRepo(env.pool)
    orgs = []
    for label in ("first", "second"):
        org = await org_repo.create_organization(
            name=f"Synthetic {label} email organization",
            owner_user_id=alice.id,
            slug=f"synthetic-{label}-email-org",
        )
        await org_repo.add_org_member(org_id=org["id"], user_id=alice.id, role="owner")
        orgs.append(org)
    first, second = orgs
    first_team = await org_repo.create_team(org_id=first["id"], name="Synthetic first-org team")
    await org_repo.add_team_member(team_id=first_team["id"], user_id=alice.id, role="member")
    await quota_repo.upsert_org_quota(first["id"], quota_mb=1024)
    await quota_repo.upsert_org_quota(second["id"], quota_mb=0)
    key = await env.manager.create_api_key(user_id=alice.id, name="two-org-write", scope="write")
    second_header = {"X-TLDW-Org-Id": str(second["id"])}
    blocked = await _upload(env, key["key"], 9010, headers=second_header)
    assert blocked.status_code == 413, blocked.text
    await quota_repo.upsert_org_quota(second["id"], quota_mb=1024)
    first_response = await _upload(env, key["key"], 9010, headers={"X-TLDW-Org-Id": str(first["id"])})
    second_response = await _upload(env, key["key"], 9010, headers=second_header)
    assert first_response.status_code == 200, first_response.text
    assert second_response.status_code == 200, second_response.text
    first_id = first_response.json()["results"][0]["db_id"]
    second_id = second_response.json()["results"][0]["db_id"]
    assert first_id != second_id
    db = MediaDatabase(db_path=str(alice.path), client_id=str(alice.id))
    try:
        second_media = db.get_media_by_id(second_id)
        assert second_media["org_id"] == second["id"]
        assert second_media["team_id"] is None
    finally:
        db.close_connection()
    message_ids = {}
    for org, expected_id in ((first, first_id), (second, second_id)):
        headers = {"X-API-KEY": key["key"], "X-TLDW-Org-Id": str(org["id"])}
        search = await env.client.get("/api/v1/email/search", headers=headers, params={"q": "subject:9010"})
        assert search.status_code == 200, search.text
        assert [item["media_id"] for item in search.json()["items"]] == [expected_id]
        message_ids[org["id"]] = search.json()["items"][0]["email_message_id"]
    for org, other_org in ((first, second), (second, first)):
        headers = {"X-API-KEY": key["key"], "X-TLDW-Org-Id": str(org["id"])}
        other_id = message_ids[other_org["id"]]
        other_detail = await env.client.get(f"/api/v1/email/messages/{other_id}", headers=headers)
        assert other_detail.status_code == 404, other_detail.text


async def test_upload_rejects_unjoined_org_selection(authenticated_email):
    env = authenticated_email
    alice, bob = env.users
    org = await AuthnzOrgsTeamsRepo(env.pool).create_organization(
        name="Synthetic Bob-only email organization",
        owner_user_id=bob.id,
        slug="synthetic-bob-only-email-org",
    )
    await AuthnzOrgsTeamsRepo(env.pool).add_org_member(org_id=org["id"], user_id=bob.id, role="owner")
    write_key = await env.manager.create_api_key(user_id=alice.id, name="org-selection-write", scope="write")
    response = await _upload(
        env,
        write_key["key"],
        9011,
        headers={"X-TLDW-Org-Id": str(org["id"])},
    )
    assert response.status_code == 403, response.text
    assert await _count(env, alice) == len(alice.message_ids)


async def test_org_scoped_key_cannot_select_another_joined_org(authenticated_email):
    env = authenticated_email
    alice = env.users[0]
    org_repo = AuthnzOrgsTeamsRepo(env.pool)
    orgs = []
    for label in ("key-first", "key-second"):
        org = await org_repo.create_organization(
            name=f"Synthetic {label} organization",
            owner_user_id=alice.id,
            slug=f"synthetic-{label}-organization",
        )
        await org_repo.add_org_member(org_id=org["id"], user_id=alice.id, role="owner")
        orgs.append(org)
    key = await env.manager.create_virtual_key(
        user_id=alice.id,
        name="org-first-write",
        org_id=orgs[0]["id"],
        scope="write",
    )
    headers = {"X-TLDW-Org-Id": str(orgs[1]["id"])}
    upload = await _upload(env, key["key"], 9012, headers=headers)
    assert upload.status_code == 403, upload.text
    search = await env.client.get(
        "/api/v1/email/search",
        headers={"X-API-KEY": key["key"], **headers},
        params={"q": "subject:9012"},
    )
    assert search.status_code == 403, search.text


async def test_active_org_jwt_upload_matches_default_search_tenant(authenticated_email):
    env = authenticated_email
    alice = env.users[0]
    org_repo = AuthnzOrgsTeamsRepo(env.pool)
    orgs = []
    for label in ("jwt-first", "jwt-second"):
        org = await org_repo.create_organization(
            name=f"Synthetic {label} organization",
            owner_user_id=alice.id,
            slug=f"synthetic-{label}-organization",
        )
        await org_repo.add_org_member(org_id=org["id"], user_id=alice.id, role="owner")
        orgs.append(org)
    await AuthnzStorageQuotasRepo(env.pool).upsert_org_quota(orgs[0]["id"], quota_mb=0)
    await AuthnzStorageQuotasRepo(env.pool).upsert_org_quota(orgs[1]["id"], quota_mb=1024)
    token = JWTService().create_access_token(
        user_id=alice.id,
        username="alice",
        role="user",
        additional_claims={
            "org_ids": [org["id"] for org in orgs],
            "active_org_id": orgs[1]["id"],
        },
    )
    headers = {"Authorization": f"Bearer {token}"}
    upload = await _upload(env, None, 9013, headers=headers)
    assert upload.status_code == 200, upload.text
    search = await env.client.get("/api/v1/email/search", headers=headers, params={"q": "subject:9013"})
    assert search.status_code == 200, search.text
    assert [item["media_id"] for item in search.json()["items"]] == [upload.json()["results"][0]["db_id"]]
