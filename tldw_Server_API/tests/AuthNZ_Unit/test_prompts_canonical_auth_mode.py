"""Prompt route mode selection, with real per-user collection databases."""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException, Request
from httpx import ASGITransport, AsyncClient

from tldw_Server_API.app.api.v1.API_Deps import Prompts_DB_Deps as db_deps
from tldw_Server_API.app.api.v1.endpoints import prompts
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
SINGLE_KEY = "synthetic-prompt-single-key"


@pytest_asyncio.fixture
async def route(monkeypatch, tmp_path):
    """Control credential resolution, retaining route policy and actual DB selection."""
    monkeypatch.delenv("APP_MODE", raising=False)
    monkeypatch.delenv("PROMPTS_REQUIRE_ADMIN", raising=False)
    canonical = Settings(_env_file=None, AUTH_MODE="multi_user",
                         JWT_SECRET_KEY="synthetic-jwt-signing-key-for-prompts-00000")
    monkeypatch.setattr(prompts, "get_auth_settings", lambda: canonical)
    monkeypatch.setitem(prompts.settings, "SINGLE_USER_MODE", True)
    monkeypatch.setitem(prompts.settings, "SINGLE_USER_API_KEY", SINGLE_KEY)
    monkeypatch.setattr(db_deps, "_get_prompts_db_path_for_user",
                        lambda user_id, salt=None: tmp_path / f"{user_id}-{salt}.sqlite")
    calls = []

    async def resolve(request: Request, api_key=None, token=None, legacy_token_header=None):
        calls.append((api_key, token, legacy_token_header))
        credential = token or api_key or (
            request.headers.get("Authorization") or request.headers.get("X-API-KEY")
            or request.headers.get("Token", "")
        ).removeprefix("Bearer ")
        users = {"alice": (2, "user"), "bob": (3, "user"), "admin": (4, "admin")}
        if credential not in users:
            raise HTTPException(401, "Invalid synthetic credential")
        user_id, role = users[credential]
        return User(id=user_id, username=credential, is_active=True, is_verified=True,
                    roles=[role], permissions=["prompts.read"])

    app = FastAPI()
    app.include_router(prompts.router, prefix="/api/v1/prompts")
    app.dependency_overrides[db_deps.get_request_user] = resolve
    monkeypatch.setattr(prompts, "get_request_user", resolve)
    async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False),
                           base_url="http://test") as client:
        yield client, canonical, calls
    await db_deps.close_all_cached_prompts_db_instances()
    await db_deps.stop_prompts_pending_close_worker()


@pytest.mark.parametrize("legacy", [True, False])
async def test_canonical_multi_collections_remain_owner_scoped(route, monkeypatch, legacy):
    client, canonical, calls = route
    assert canonical.AUTH_MODE == "multi_user"
    monkeypatch.setitem(prompts.settings, "SINGLE_USER_MODE", legacy)
    alice = {"Authorization": "Bearer alice"}
    bob = {"Authorization": "Bearer bob"}
    created = await client.post("/api/v1/prompts/collections/create", headers=alice,
                                json={"name": "Alice private collection"})
    assert created.status_code == 200, created.text
    own = await client.get("/api/v1/prompts/collections", headers=alice)
    foreign = await client.get("/api/v1/prompts/collections", headers=bob)
    assert own.status_code == foreign.status_code == 200
    assert [item["name"] for item in own.json()["collections"]] == ["Alice private collection"]
    assert foreign.json()["collections"] == []
    assert any(call[1] == "alice" for call in calls)


@pytest.mark.parametrize("headers,expected", [
    ({}, 401), ({"Authorization": "Bearer invalid"}, 401),
    ({"X-API-KEY": "invalid"}, 401),
    ({"X-API-KEY": "alice"}, 200),
    ({"Token": "Bearer alice"}, 200),
])
async def test_multi_credentials_delegate_without_single_user_bypass(route, headers, expected):
    response = await route[0].get("/api/v1/prompts/collections", headers=headers)
    assert response.status_code == expected


async def test_multi_admin_policy_stays_claim_based(route, monkeypatch):
    monkeypatch.setenv("PROMPTS_REQUIRE_ADMIN", "true")
    ordinary = await route[0].get("/api/v1/prompts/collections", headers={"Authorization": "Bearer alice"})
    admin = await route[0].get("/api/v1/prompts/collections", headers={"Authorization": "Bearer admin"})
    assert ordinary.status_code == 403
    assert admin.status_code == 200


@pytest.mark.parametrize("credential,expected", [(SINGLE_KEY, 200), ("wrong", 401), (None, 401)])
async def test_canonical_single_ignores_legacy_false(route, monkeypatch, credential, expected):
    monkeypatch.setattr(prompts, "get_auth_settings", lambda: SimpleNamespace(AUTH_MODE="single_user"))
    monkeypatch.setitem(prompts.settings, "SINGLE_USER_MODE", False)
    # Database identity is separately resolved by the existing shared dependency.
    headers = {"Authorization": "Bearer alice"}
    if credential is not None:
        headers["Token"] = credential
    else:
        headers = {}
    response = await route[0].get("/api/v1/prompts/collections", headers=headers)
    assert response.status_code == expected


async def test_canonical_settings_failure_cannot_authorize_legacy_single_admin(route, monkeypatch):
    def fail_settings():
        raise ValueError("Invalid canonical settings")

    monkeypatch.setattr(prompts, "get_auth_settings", fail_settings)
    response = await route[0].get("/api/v1/prompts/collections", headers={
        "Token": SINGLE_KEY, "Authorization": "Bearer alice"})
    assert response.status_code == 500
