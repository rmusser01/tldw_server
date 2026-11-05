import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_single_user_rbac_seed_sqlite_file_applies_schema_and_seeds(tmp_path, monkeypatch, fresh_services):
    """ensure_single_user_rbac_seed_if_needed should succeed on a fresh SQLite DB file.

    This specifically exercises the "no inline CREATE TABLE in the seed path" behavior for
    file-backed SQLite, relying on migrations/schema backstops instead.
    """
    db_path = tmp_path / "authnz_users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    reset_settings()

    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_single_user_rbac_seed_if_needed

        await ensure_single_user_rbac_seed_if_needed()

        conn = sqlite3.connect(str(db_path))
        try:
            assert conn.execute("SELECT 1 FROM roles WHERE name = 'admin'").fetchone() is not None
            assert conn.execute("SELECT 1 FROM users WHERE id = 1").fetchone() is not None
        finally:
            conn.close()
    finally:
        reset_settings()


@pytest.mark.asyncio
async def test_single_user_rbac_seed_does_not_store_the_test_key_in_test_mode(tmp_path, monkeypatch, fresh_services):
    """TEST_MODE seeding stores only the configured key; the test key never authenticates."""
    from fastapi import HTTPException
    from starlette.requests import Request

    db_path = tmp_path / "authnz_users.db"
    configured = "configured-key-1234567890"
    test_key = "other-test-key-0987654321"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setenv("SINGLE_USER_API_KEY", configured)
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", test_key)

    reset_settings()

    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_single_user_rbac_seed_if_needed
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import authenticate_api_key_user

        await ensure_single_user_rbac_seed_if_needed()

        conn = sqlite3.connect(str(db_path))
        try:
            stored = conn.execute("SELECT COUNT(*) FROM api_keys WHERE name = 'single-user test key'").fetchone()[0]
        finally:
            conn.close()
        assert stored == 0

        def _request() -> Request:
            return Request({"type": "http", "method": "GET", "path": "/t", "headers": [], "client": ("127.0.0.1", 0)})

        with pytest.raises(HTTPException) as exc:
            await authenticate_api_key_user(_request(), test_key)
        assert exc.value.status_code == 401

        user = await authenticate_api_key_user(_request(), configured)
        assert user.username == "single_user"
    finally:
        reset_settings()


@pytest.mark.asyncio
async def test_single_user_rbac_seed_sqlite_file_backstops_rbac_tables_when_migrations_skip(
    tmp_path,
    monkeypatch, fresh_services
):
    """File-backed SQLite seed should tolerate a skipped migration-lock path."""
    db_path = tmp_path / "authnz_users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    def _raise_migration_lock_unavailable(_db_path):
        raise RuntimeError("Redis unavailable for migration lock; refusing local file fallback")

    reset_settings()

    try:
        from tldw_Server_API.app.core.AuthNZ import initialize

        monkeypatch.setattr(initialize, "ensure_authnz_tables", _raise_migration_lock_unavailable)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.AuthNZ.database.ensure_authnz_tables",
            _raise_migration_lock_unavailable,
        )
        initialize._SCHEMA_ENSURED_KEYS.clear()

        await initialize.ensure_single_user_rbac_seed_if_needed()

        conn = sqlite3.connect(str(db_path))
        try:
            assert conn.execute("SELECT 1 FROM roles WHERE name = 'admin'").fetchone() is not None
            assert conn.execute("SELECT 1 FROM users WHERE id = 1").fetchone() is not None
        finally:
            conn.close()
    finally:
        reset_settings()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_startup_seed_revokes_a_stale_test_key_row_stored_by_an_older_boot(
    backend, tmp_path, monkeypatch, request, fresh_services
):
    """A 'single-user test key' admin row from an older TEST_MODE boot stops authenticating."""
    from fastapi import HTTPException
    from starlette.requests import Request

    configured = "configured-key-1234567890"
    stale = "old-test-key-seeded-by-v0-0987654321"
    if backend == "postgres":
        dsn = request.getfixturevalue("pg_temp_db")["dsn"]
        monkeypatch.setenv("DATABASE_URL", dsn)
    else:
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'authnz_users.db'}")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", configured)
    for name in ("TEST_MODE", "TLDW_TEST_MODE", "TESTING", "SINGLE_USER_TEST_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MCP_SINGLE_USER_COMPAT_SHIM", "1")

    reset_settings()
    try:
        from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint as mcp_ep
        from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_single_user_rbac_seed_if_needed
        from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import authenticate_api_key_user
        from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

        pool = await get_db_pool()
        # Postgres has no schema until the stores initialize; SQLite gets it from the seed.
        await UsersDB(pool).initialize(ensure_schema=False)
        manager = APIKeyManager(db_pool=pool)
        await manager.initialize()
        await ensure_single_user_rbac_seed_if_needed()
        # The row the pre-TASK-13950.19 seed stored under TEST_MODE.
        await AuthnzApiKeysRepo(pool).upsert_primary_key(
            user_id=1,
            key_hash=manager.hash_api_key(stale),
            key_identifier=None,
            key_prefix=stale[:10] + "...",
            name="single-user test key",
            description="Deterministic API key for test automation",
            scope="admin",
            is_virtual=False,
        )
        assert await manager.validate_api_key(stale, ip_address="127.0.0.1")

        await ensure_single_user_rbac_seed_if_needed()
        await ensure_single_user_rbac_seed_if_needed()  # idempotent

        def _req() -> Request:
            return Request({"type": "http", "method": "GET", "path": "/t", "headers": [], "client": ("127.0.0.1", 0)})

        assert await manager.validate_api_key(stale, ip_address="127.0.0.1") is None
        assert await mcp_ep._resolve_token_data_compat(credentials=None, x_api_key=stale, request=_req()) is None
        with pytest.raises(HTTPException) as exc:
            await authenticate_api_key_user(_req(), stale)
        assert exc.value.status_code == 401

        user = await authenticate_api_key_user(_req(), configured)
        assert user.username == "single_user"
        token = await mcp_ep._resolve_token_data_compat(credentials=None, x_api_key=configured, request=_req())
        assert token is not None and token.permissions == []
    finally:
        reset_settings()
