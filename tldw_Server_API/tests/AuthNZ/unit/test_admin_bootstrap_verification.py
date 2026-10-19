"""Explicit admin bootstrap verification through the canonical SQLite stores."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.users import _require_active_verified_user
from tldw_Server_API.app.core.AuthNZ import api_key_manager, create_admin, initialize
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.rbac_seed import ensure_baseline_rbac_seed
from tldw_Server_API.app.core.AuthNZ.repos import users_repo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_USERNAME = "verified-bootstrap-fixture"
_EMAIL = "bootstrap-fixture@example.com"
_PASSWORD = "BootstrapFixture-742!"  # nosec B105 -- synthetic test password, never a runtime credential.


@pytest.fixture
async def owned_auth(tmp_path, monkeypatch):
    """Bind production bootstrap repositories to one normally initialized DB."""
    pool = DatabasePool(Settings(AUTH_MODE="multi_user", DATABASE_URL=f"sqlite:///{tmp_path / 'auth.db'}"))
    await pool.initialize()
    async with pool.transaction() as connection:
        await ensure_baseline_rbac_seed(connection, is_postgres=False)
        await connection.commit()
    users = UsersDB(pool)
    await users.initialize(ensure_schema=False)
    repo = AuthnzUsersRepo(pool)
    monkeypatch.setattr(users_repo, "get_db_pool", AsyncMock(return_value=pool))
    manager = SimpleNamespace(create_api_key=AsyncMock(return_value={"key": "fixture-key"}))
    monkeypatch.setattr(api_key_manager, "get_api_key_manager", AsyncMock(return_value=manager))
    for module in (create_admin, initialize):
        monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(AUTH_MODE="multi_user"))
        monkeypatch.setattr(module, "get_users_db", AsyncMock(return_value=users))
        monkeypatch.setattr(module, "AuthnzUsersRepo", AuthnzUsersRepo)
        monkeypatch.setattr(module, "ensure_user_directories", AsyncMock())
    monkeypatch.setattr(initialize, "get_api_key_manager", AsyncMock(return_value=manager))
    monkeypatch.setattr("builtins.input", lambda prompt: _USERNAME if "username" in prompt else _EMAIL)
    monkeypatch.setattr(initialize, "getpass", lambda _prompt: _PASSWORD)
    try:
        yield users, repo
    finally:
        await pool.close()


async def _bootstrap(entrypoint: str) -> bool:
    if entrypoint == "interactive":
        return await initialize.create_admin_user()
    return await create_admin.create_admin_user_non_interactive(_USERNAME, _PASSWORD, _EMAIL)


async def _existing_user(users, *, role="admin", verified=False, active=False, superuser=False):
    password_hash = PasswordService().hash_password(_PASSWORD)
    return await users.create_user(
        username=_USERNAME,
        email=_EMAIL,
        password_hash=password_hash,
        role=role,
        is_verified=verified,
        is_active=active,
        is_superuser=superuser,
    )


@pytest.mark.parametrize("entrypoint", ["interactive", "noninteractive"])
@pytest.mark.parametrize("existing", ["new", "unverified", "verified"])
async def test_explicit_bootstrap_verifies_owned_admin_and_profile_after_membership(
    owned_auth, monkeypatch, entrypoint, existing
):
    users, repo = owned_auth
    seeded = None if existing == "new" else await _existing_user(users, verified=existing == "verified")
    membership_at_activation = []
    original_update = users.update_user

    async def update_after_membership(user_id, **fields):
        if fields.get("is_active"):
            before = await users.get_user_by_id(user_id)
            membership_at_activation.append(
                (await repo.has_role_assignment(user_id=user_id, role_name="admin"), before["is_active"])
            )
        return await original_update(user_id, **fields)

    monkeypatch.setattr(users, "update_user", update_after_membership)
    assert await _bootstrap(entrypoint) is True  # nosec B101 -- pytest regression oracle.
    first = await users.get_user_by_username(_USERNAME)
    assert first["is_verified"] is True  # nosec B101 -- pytest regression oracle.
    assert first["is_active"] is True and first["is_superuser"] is True  # nosec B101 -- pytest regression oracle.
    _require_active_verified_user(first)
    assert membership_at_activation == [(True, False)]  # nosec B101 -- pytest regression oracle.
    assert PasswordService().verify_password(_PASSWORD, first["password_hash"])[0] is True  # nosec B101 -- pytest regression oracle.
    if seeded:
        assert first["id"] == seeded["id"] and first["password_hash"] == seeded["password_hash"]  # nosec B101 -- pytest regression oracle.
    assert await _bootstrap(entrypoint) is True  # nosec B101 -- pytest regression oracle.
    second = await users.get_user_by_username(_USERNAME)
    assert second["id"] == first["id"] and second["password_hash"] == first["password_hash"]  # nosec B101 -- pytest regression oracle.
    assert second["email"] == first["email"] and second["is_verified"] is True  # nosec B101 -- pytest regression oracle.
    assert await repo.has_role_assignment(user_id=second["id"], role_name="admin") is True  # nosec B101 -- pytest regression oracle.
    _require_active_verified_user(second)


async def test_ordinary_users_db_default_remains_unverified_and_profile_refuses(owned_auth):
    users, _repo = owned_auth
    password_hash = PasswordService().hash_password(_PASSWORD)
    ordinary = await users.create_user(username=_USERNAME, email=_EMAIL, password_hash=password_hash)
    assert ordinary["is_verified"] is False  # nosec B101 -- pytest regression oracle.
    with pytest.raises(HTTPException) as refused:
        _require_active_verified_user(ordinary)
    assert refused.value.status_code == 403 and refused.value.detail == "Email verification required"  # nosec B101 -- pytest regression oracle.


@pytest.mark.parametrize("entrypoint", ["interactive", "noninteractive"])
async def test_explicit_bootstrap_refuses_existing_ordinary_user_without_changes(owned_auth, entrypoint):
    users, repo = owned_auth
    original = await _existing_user(users, role="user", active=True)
    assert await _bootstrap(entrypoint) is False  # nosec B101 -- pytest regression oracle.
    assert await users.get_user_by_id(original["id"]) == original  # nosec B101 -- pytest regression oracle.
    assert await repo.has_role_assignment(user_id=original["id"], role_name="admin") is False  # nosec B101 -- pytest regression oracle.


@pytest.mark.parametrize("entrypoint", ["interactive", "noninteractive"])
@pytest.mark.parametrize("existing", [False, True])
async def test_membership_failure_leaves_admin_disabled_and_unverified(owned_auth, monkeypatch, entrypoint, existing):
    users, repo = owned_auth
    if existing:
        await _existing_user(users, active=True, superuser=True)
    monkeypatch.setattr(
        AuthnzUsersRepo, "assign_role_if_missing", AsyncMock(side_effect=RuntimeError("fixture membership unavailable"))
    )
    assert await _bootstrap(entrypoint) is False  # nosec B101 -- pytest regression oracle.
    row = await users.get_user_by_username(_USERNAME)
    assert row["is_verified"] is False and row["is_active"] is False and row["is_superuser"] is False  # nosec B101 -- pytest regression oracle.
    assert await repo.has_role_assignment(user_id=row["id"], role_name="admin") is False  # nosec B101 -- pytest regression oracle.
    with pytest.raises(HTTPException, match="inactive"):
        _require_active_verified_user(row)


@pytest.mark.parametrize("entrypoint", ["interactive", "noninteractive"])
async def test_activation_write_failure_does_not_verify_or_enable_admin(owned_auth, monkeypatch, entrypoint):
    users, repo = owned_auth
    seeded = await _existing_user(users, active=True, superuser=True)
    original_update = users.update_user

    async def fail_activation(user_id, **fields):
        if fields.get("is_active"):
            raise RuntimeError("fixture activation write failed")
        return await original_update(user_id, **fields)

    monkeypatch.setattr(users, "update_user", fail_activation)
    assert await _bootstrap(entrypoint) is False  # nosec B101 -- pytest regression oracle.
    row = await users.get_user_by_id(seeded["id"])
    assert row["is_verified"] is False and row["is_active"] is False and row["is_superuser"] is False  # nosec B101 -- pytest regression oracle.
    assert await repo.has_role_assignment(user_id=row["id"], role_name="admin") is True  # nosec B101 -- pytest regression oracle.


async def test_generic_admin_membership_scan_does_not_verify_or_activate(owned_auth):
    users, repo = owned_auth
    original = await _existing_user(users)
    await initialize._repair_existing_admin_memberships(users)
    assert await users.get_user_by_id(original["id"]) == original  # nosec B101 -- pytest regression oracle.
    assert await repo.has_role_assignment(user_id=original["id"], role_name="admin") is True  # nosec B101 -- pytest regression oracle.
