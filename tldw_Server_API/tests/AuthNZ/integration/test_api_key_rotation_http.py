"""Public rotation error mapping with real managers and persisted key state."""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import users as users_endpoint
from tldw_Server_API.app.api.v1.endpoints.admin import admin_api_keys
from tldw_Server_API.app.api.v1.utils.exception_handlers import global_unhandled_exception_handler
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.AuthNZ import api_key_manager as manager_module
from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
from tldw_Server_API.app.services import admin_api_keys_service
from tldw_Server_API.tests.AuthNZ.integration import test_api_key_rotation_boundaries as rotation_boundaries

pytestmark = pytest.mark.integration
rotation_context = rotation_boundaries.rotation_context


@pytest_asyncio.fixture(params=["self", "admin"])
async def rotation_http_context(
    request: pytest.FixtureRequest,
    rotation_context: tuple[APIKeyManager, int],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastAPI, APIKeyManager, int, str]:
    manager, user_id = rotation_context
    users = UsersDB(manager.db_pool)
    await users.initialize(ensure_schema=False)
    await users.update_user(user_id, is_verified=True)
    principal = AuthPrincipal(
        kind="user",
        user_id=user_id,
        username="rotation-user",
        roles=["admin"] if request.param == "admin" else ["user"],
        is_admin=request.param == "admin",
    )
    app = FastAPI()
    app.include_router(users_endpoint.router, prefix="/api/v1")
    app.include_router(admin_api_keys.router, prefix="/api/v1/admin")
    app.add_exception_handler(Exception, global_unhandled_exception_handler)
    app.dependency_overrides[get_auth_principal] = lambda: principal
    monkeypatch.setattr(users_endpoint, "get_api_key_manager", AsyncMock(return_value=manager))
    monkeypatch.setattr(admin_api_keys_service, "get_api_key_manager", AsyncMock(return_value=manager))
    prefix = "/api/v1/users" if request.param == "self" else f"/api/v1/admin/users/{user_id}"
    return app, manager, user_id, prefix


@pytest.mark.asyncio
@pytest.mark.parametrize("source_state", ["missing", "foreign", "revoked"])
async def test_rotation_routes_return_same_not_found_without_mutation(
    rotation_http_context: tuple[FastAPI, APIKeyManager, int, str],
    source_state: str,
) -> None:
    app, manager, user_id, prefix = rotation_http_context
    owner_id = user_id
    if source_state == "foreign":
        users = UsersDB(manager.db_pool)
        await users.initialize(ensure_schema=False)
        owner = await users.create_user(  # nosec B106 # Inert fixture hash, never used for login
            username="other-owner", email="other@example.com", password_hash="unused-fixture-hash"
        )
        owner_id = int(owner["id"])
    created = await manager.create_api_key(user_id=owner_id, name="source", scope="read")
    key_id = int(created["id"])
    if source_state == "revoked":
        await manager.revoke_api_key(key_id, user_id=user_id, reason="test rejection")
    if source_state == "missing":
        key_id += 1
    before = [dict(row) for row in await manager.db_pool.fetchall("SELECT * FROM api_keys ORDER BY id")]

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.post(f"{prefix}/api-keys/{key_id}/rotate", json={})

    assert response.status_code == 404
    assert response.json() == {"detail": "API key not found"}
    assert [dict(row) for row in await manager.db_pool.fetchall("SELECT * FROM api_keys ORDER BY id")] == before


@pytest.mark.asyncio
async def test_rotation_routes_preserve_audit_failure_and_rollback(
    rotation_http_context: tuple[FastAPI, APIKeyManager, int, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, manager, user_id, prefix = rotation_http_context
    created = await manager.create_api_key(user_id=user_id, name="source", scope="read")
    before = [dict(row) for row in await manager.db_pool.fetchall("SELECT * FROM api_keys ORDER BY id")]
    monkeypatch.setattr(
        manager_module,
        "emit_mandatory_api_key_management_audit",
        AsyncMock(side_effect=MandatoryAuditWriteError("Mandatory audit persistence unavailable")),
    )

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.post(f"{prefix}/api-keys/{created['id']}/rotate", json={})

    assert response.status_code == 503
    assert response.json() == {"detail": "Mandatory audit persistence unavailable"}
    assert [dict(row) for row in await manager.db_pool.fetchall("SELECT * FROM api_keys ORDER BY id")] == before


@pytest.mark.asyncio
async def test_rotation_routes_do_not_map_unrelated_value_errors_to_not_found(
    rotation_http_context: tuple[FastAPI, APIKeyManager, int, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, manager, _user_id, prefix = rotation_http_context
    monkeypatch.setattr(
        manager._get_repo(), "fetch_key_for_user", AsyncMock(side_effect=ValueError("private-storage-sentinel"))
    )

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.post(f"{prefix}/api-keys/1/rotate", json={})

    assert response.status_code == 500
    assert "private-storage-sentinel" not in response.text
