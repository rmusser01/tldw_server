import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_single_user_rbac_seed_sqlite_file_applies_schema_and_seeds(tmp_path, monkeypatch):
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
    await reset_db_pool()

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
        await reset_db_pool()
        reset_settings()


@pytest.mark.asyncio
async def test_single_user_rbac_seed_sqlite_file_backstops_rbac_tables_when_migrations_skip(
    tmp_path,
    monkeypatch,
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
    await reset_db_pool()

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
        await reset_db_pool()
        reset_settings()
