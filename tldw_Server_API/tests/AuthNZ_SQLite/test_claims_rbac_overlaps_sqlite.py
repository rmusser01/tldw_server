import os
from pathlib import Path

import pytest
from starlette.requests import Request
from starlette.types import Scope

from tldw_Server_API.app.core.AuthNZ.auth_principal_resolver import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


def _base_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-rbac-claims-1234567890")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("VIRTUAL_KEYS_ENABLED", "false")
    monkeypatch.setenv("LLM_BUDGET_ENFORCE", "false")


def _make_request(api_key: str) -> Request:
    scope: Scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/claims-test",
        "headers": [(b"x-api-key", api_key.encode("latin-1"))],
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


async def _seed_user(pool, username: str) -> int:
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active, is_verified, role)
            VALUES (?, ?, ?, 1, 1, 'user')
            """,
            (username, f"{username}@example.com", "x"),
        )
    return int(await pool.fetchval("SELECT id FROM users WHERE username = ?", username))


async def _ensure_role(pool, role_name: str) -> int:
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO roles (name, description, is_system) VALUES (?, ?, 0)",
            (role_name, role_name),
        )
    row = await pool.fetchrow("SELECT id FROM roles WHERE name = ?", role_name)
    return int(row["id"] if isinstance(row, dict) else row[0])


async def _ensure_permission(pool, code: str) -> int:
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO permissions (name, description, category) VALUES (?, ?, ?)",
            (code, code, code.split(".")[0] if "." in code else "general"),
        )
    row = await pool.fetchrow("SELECT id FROM permissions WHERE name = ?", code)
    return int(row["id"] if isinstance(row, dict) else row[0])


async def _attach_role_permission(pool, role_id: int, perm_id: int) -> None:
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
            (role_id, perm_id),
        )


async def _assign_user_role(pool, user_id: int, role_id: int) -> None:
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
            (user_id, role_id),
        )


async def _set_user_override(pool, user_id: int, perm_id: int, granted: bool) -> None:
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT OR REPLACE INTO user_permissions (user_id, permission_id, granted)
            VALUES (?, ?, ?)
            """,
            (user_id, perm_id, 1 if granted else 0),
        )


async def _create_api_key(user_id: int) -> str:
    mgr = APIKeyManager()
    await mgr.initialize()
    key_info = await mgr.create_api_key(
        user_id=user_id,
        name="claims-test-key",
        description="rbac claims sqlite",
        scope="read",
        expires_in_days=7,
    )
    return str(key_info["key"])


@pytest.mark.asyncio
async def test_claims_include_overlapping_role_permissions_sqlite(tmp_path, monkeypatch):
    _base_env(tmp_path, monkeypatch)
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    user_id = await _seed_user(pool, "overlap_user")
    role_alpha = await _ensure_role(pool, "alpha")
    role_beta = await _ensure_role(pool, "beta")

    perm_shared = await _ensure_permission(pool, "perm.shared")
    perm_alpha = await _ensure_permission(pool, "perm.alpha")
    perm_beta = await _ensure_permission(pool, "perm.beta")

    await _attach_role_permission(pool, role_alpha, perm_shared)
    await _attach_role_permission(pool, role_alpha, perm_alpha)
    await _attach_role_permission(pool, role_beta, perm_shared)
    await _attach_role_permission(pool, role_beta, perm_beta)

    await _assign_user_role(pool, user_id, role_alpha)
    await _assign_user_role(pool, user_id, role_beta)

    api_key = await _create_api_key(user_id)
    principal = await get_auth_principal(_make_request(api_key))

    perms = list(principal.permissions or [])
    assert {"perm.shared", "perm.alpha", "perm.beta"}.issubset(set(perms))
    assert len(perms) == len(set(perms))


@pytest.mark.asyncio
async def test_claims_reflect_user_allow_deny_overrides_sqlite(tmp_path, monkeypatch):
    _base_env(tmp_path, monkeypatch)
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    user_id = await _seed_user(pool, "override_user")
    role_base = await _ensure_role(pool, "base")

    perm_base = await _ensure_permission(pool, "perm.base")
    perm_allow = await _ensure_permission(pool, "perm.override.allow")

    await _attach_role_permission(pool, role_base, perm_base)
    await _assign_user_role(pool, user_id, role_base)

    await _set_user_override(pool, user_id, perm_allow, granted=True)
    await _set_user_override(pool, user_id, perm_base, granted=False)

    api_key = await _create_api_key(user_id)
    principal = await get_auth_principal(_make_request(api_key))

    perms = set(principal.permissions or [])
    assert "perm.override.allow" in perms
    assert "perm.base" not in perms
