"""Real API-key authentication and per-user SQLite email routing.

Users/keys are created through AuthNZ services. Authentication and Media database
dependencies are not overridden. This does not certify deployed infrastructure,
password/JWT login, upload quotas, or PostgreSQL RLS.
"""

import asyncio
import socket
from contextlib import nullcontext
from secrets import token_urlsafe
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from tldw_Server_API.app.api.v1.API_Deps import DB_Deps
from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import shutdown_all_audit_services
from tldw_Server_API.app.api.v1.endpoints import email as email_endpoint
from tldw_Server_API.app.api.v1.endpoints.media import add, listing
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager, reset_api_key_manager
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.Billing.enforcement import reset_billing_enforcer
from tldw_Server_API.app.core.Billing.subscription_service import reset_subscription_service
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.services.storage_quota_service import reset_storage_service
from tldw_Server_API.tests.DB_Management.test_email_search_cursor import add_message

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def authenticated_email(tmp_path, monkeypatch, request):
    """Use real isolated services and reject even swallowed outbound attempts."""
    from fastapi import BackgroundTasks

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Chunking.auto_boundary_assistant import ChatAutoChunkBoundaryAssistant
    from tldw_Server_API.app.core.Claims_Extraction import claims_utils
    from tldw_Server_API.app.core.Embeddings.jobs_adapter import EmbeddingsJobsAdapter
    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib

    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append("outbound or model call")
        raise AssertionError("Authenticated email validation must stay offline")

    for target, name in [
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
        (socket, "getaddrinfo"),
        (Summarization_General_Lib, "analyze"),
        (EmbeddingsJobsAdapter, "create_job"),
        (claims_utils, "extract_claims_for_chunks"),
        (ChatAutoChunkBoundaryAssistant, "refine"),
        (BackgroundTasks, "add_task"),
    ]:
        monkeypatch.setattr(target, name, forbidden)
    for name in ("fetch", "afetch", "apost", "fetch_json", "afetch_json", "download", "adownload"):
        monkeypatch.setattr(http_client, name, forbidden)

    for name, value in {
        "AUTH_MODE": "multi_user",
        "PROFILE": "multi-user-sqlite",
        "DATABASE_URL": f"sqlite:///{tmp_path / 'auth.sqlite'}",
        "USER_DB_BASE_DIR": str(tmp_path / "users"),
        "CONTENT_DB_MODE": "sqlite",
        "JWT_SECRET_KEY": token_urlsafe(48),
        "REDIS_URL": "",
        "TEST_MODE": "true",
        "TESTING": "false",
        "EVALS_HEAVY_ADMIN_ONLY": "true",
        "tldw_production": "false",
        "CONNECTORS_WORKER_ENABLED": "false",
        "DEFER_HEAVY_STARTUP": "true",
        "STORAGE_QUOTA_ENFORCEMENT": "1",
        "STORAGE_QUOTA_FAIL_OPEN": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", True)
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_GMAIL_CONNECTOR_ENABLED", False)
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_NATIVE_PERSIST_ENABLED", True)
    monkeypatch.setitem(listing.settings, "EMAIL_MEDIA_SEARCH_DELEGATION_MODE", "opt_in")

    async def reset():
        await shutdown_all_audit_services()
        await reset_storage_service()
        await reset_subscription_service()
        reset_billing_enforcer()
        await reset_api_key_manager()
        await reset_db_pool()
        reset_settings()
        DB_Deps.reset_media_db_cache()

    await reset()
    try:
        pool = await get_db_pool()
        repo = AuthnzUsersRepo(pool)
        manager = await get_api_key_manager()
        users = []
        for name, count in [("alice", 3), ("bob", 1)]:
            user_id = await repo.create_user(
                username=name,
                email=f"{name}@example.test",
                password_hash=token_urlsafe(32),
                is_verified=True,
            )
            await repo.assign_role_if_missing(user_id=user_id, role_name="user")
            key = await manager.create_api_key(user_id=user_id, name="synthetic-read", scope="read")
            path = DatabasePaths.get_media_db_path(user_id)
            assert path.is_relative_to(tmp_path)
            db = MediaDatabase(db_path=str(path), client_id=str(user_id))
            try:
                with scoped_context(user_id=user_id, org_ids=[], team_ids=[]):
                    ids = [add_message(db, f"{name}-{i}", subject=f"Synthetic {name}")[0] for i in range(count)]
            finally:
                db.close_connection()
            users.append(SimpleNamespace(id=user_id, key=key, message_ids=ids, path=path))

        from tldw_Server_API.tests.helpers.app_main_state import app_main_isolated, reload_app_main

        use_main = getattr(request, "param", None) in {"main", "main_lifespan"}
        with app_main_isolated() if use_main else nullcontext():
            if use_main:
                app = reload_app_main().app
            else:
                app = FastAPI()
                app.include_router(email_endpoint.router, prefix="/api/v1/email")
                app.include_router(listing.router, prefix="/api/v1/media")
                app.include_router(add.router, prefix="/api/v1/media")
            assert app.dependency_overrides == {}
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                yield SimpleNamespace(app=app, client=client, users=users, manager=manager, pool=pool)
    finally:
        await reset()
        assert attempts == [], f"Forbidden attempts, including caught errors: {attempts}"


@pytest.mark.parametrize("path", ["/api/v1/email/search", "/api/v1/email/messages/1"])
@pytest.mark.parametrize("credential", [None, "invalid"])
async def test_email_rejects_missing_and_invalid_credentials(authenticated_email, path, credential):
    headers = {} if credential is None else {"X-API-KEY": token_urlsafe(32)}
    response = await authenticated_email.client.get(path, headers=headers)
    assert response.status_code == 401, response.text


async def test_interleaved_users_only_read_own_messages_and_cursors(authenticated_email):
    env = authenticated_email
    alice, bob = env.users
    assert alice.path != bob.path
    pages = []
    for user, name in [(alice, "alice"), (bob, "bob"), (alice, "alice")]:
        response = await env.client.get(
            "/api/v1/email/search", params={"cursor": "", "limit": 1}, headers={"X-API-KEY": user.key["key"]}
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["items"][0]["subject"] == f"Synthetic {name}"
        assert data["pagination"]["total"] == len(user.message_ids)
        pages.append(data)
    foreign = await env.client.get(
        f"/api/v1/email/messages/{alice.message_ids[-1]}", headers={"X-API-KEY": bob.key["key"]}
    )
    assert foreign.status_code == 404, foreign.text
    cursor = pages[0]["next_cursor"]
    assert cursor
    response = await env.client.get(
        "/api/v1/email/search", params={"cursor": cursor}, headers={"X-API-KEY": bob.key["key"]}
    )
    assert response.status_code == 400, response.text
    own = await env.client.get(
        f"/api/v1/email/messages/{bob.message_ids[0]}", headers={"Authorization": f"Bearer {bob.key['key']}"}
    )
    assert own.status_code == 200, own.text
    assert own.json()["subject"] == "Synthetic bob"


async def test_concurrent_searches_keep_credential_scope(authenticated_email):
    env = authenticated_email
    users = env.users * 4
    responses = await asyncio.gather(
        *[
            env.client.get(
                "/api/v1/email/search", headers={"X-API-KEY": user.key["key"]}, params={"tenant_id": env.users[0].id}
            )
            for user in users
        ]
    )
    for user, response in zip(users, responses, strict=True):
        assert response.status_code == 200, response.text
        items = response.json()["items"]
        assert len(items) == len(user.message_ids)
        assert {row["subject"] for row in items} == {"Synthetic alice" if user == env.users[0] else "Synthetic bob"}


async def test_single_user_configured_key_and_bearer_access(authenticated_email, monkeypatch):
    env = authenticated_email
    configured_key = token_urlsafe(32)
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("PROFILE", "local-single-user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", configured_key)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", str(env.users[0].id))
    reset_settings()
    for headers in [{"X-API-KEY": configured_key}, {"Authorization": f"Bearer {configured_key}"}]:
        response = await env.client.get("/api/v1/email/search", headers=headers)
        assert response.status_code == 200, response.text
        assert {row["subject"] for row in response.json()["items"]} == {"Synthetic alice"}
    invalid = await env.client.get("/api/v1/email/search", headers={"X-API-KEY": token_urlsafe(32)})
    assert invalid.status_code == 401, invalid.text


async def test_revoked_key_cannot_read_email(authenticated_email):
    env = authenticated_email
    alice = env.users[0]
    headers = {"X-API-KEY": alice.key["key"]}
    assert (await env.client.get("/api/v1/email/search", headers=headers)).status_code == 200
    await env.manager.revoke_api_key(alice.key["id"], user_id=alice.id, reason="synthetic validation")
    assert (await env.client.get("/api/v1/email/search", headers=headers)).status_code == 401


async def test_authenticated_feature_flags_and_media_delegation(authenticated_email, monkeypatch):
    env = authenticated_email
    headers = {"X-API-KEY": env.users[1].key["key"]}
    response = await env.client.post(
        "/api/v1/media/search",
        headers=headers,
        json={"query": "subject:Synthetic", "media_types": ["email"], "email_query_mode": "operators"},
    )
    assert response.status_code == 200, response.text
    assert "Synthetic bob" in response.text
    assert "Synthetic alice" not in response.text
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", False)
    assert (await env.client.get("/api/v1/email/search", headers=headers)).status_code == 404
    assert (await env.client.get("/api/v1/email/messages/1", headers=headers)).status_code == 404
    assert (await env.client.get("/api/v1/email/sources", headers=headers)).status_code == 404


@pytest.mark.parametrize("authenticated_email", ["main"], indirect=True)
async def test_main_app_middleware_auth_and_email_route_registration(authenticated_email):
    """Exercise registered routes and middleware; ASGITransport does not run lifespan."""
    env = authenticated_email
    assert (await env.client.get("/api/v1/email/search")).status_code == 401
    headers = {"X-API-KEY": env.users[1].key["key"]}
    response = await env.client.get("/api/v1/email/search", headers=headers)
    assert response.status_code == 200, response.text
    assert [row["subject"] for row in response.json()["items"]] == ["Synthetic bob"]
    response = await env.client.get(f"/api/v1/email/messages/{env.users[0].message_ids[-1]}", headers=headers)
    assert response.status_code == 404, response.text
    response = await env.client.post(
        "/api/v1/media/search",
        headers=headers,
        json={"query": "subject:Synthetic", "media_types": ["email"], "email_query_mode": "operators"},
    )
    assert response.status_code == 200, response.text
    assert [item["title"] for item in response.json()["items"]] == ["Synthetic bob"]


@pytest.mark.parametrize("authenticated_email", ["main_lifespan"], indirect=True)
async def test_main_app_lifespan_serves_scoped_email_and_media_routes(authenticated_email):
    """Exercise the actual startup/shutdown sequence with synthetic mail and no outbound calls."""
    env = authenticated_email
    bob = env.users[1]
    headers = {"X-API-KEY": bob.key["key"]}
    async with env.app.router.lifespan_context(env.app):
        search = await env.client.get("/api/v1/email/search", headers=headers)
        assert search.status_code == 200, search.text
        assert [row["subject"] for row in search.json()["items"]] == ["Synthetic bob"]

        detail = await env.client.get(f"/api/v1/email/messages/{bob.message_ids[0]}", headers=headers)
        assert detail.status_code == 200, detail.text
        assert detail.json()["subject"] == "Synthetic bob"

        media = await env.client.post(
            "/api/v1/media/search",
            headers=headers,
            json={"query": "subject:Synthetic", "media_types": ["email"], "email_query_mode": "operators"},
        )
        assert media.status_code == 200, media.text
        assert [item["title"] for item in media.json()["items"]] == ["Synthetic bob"]
