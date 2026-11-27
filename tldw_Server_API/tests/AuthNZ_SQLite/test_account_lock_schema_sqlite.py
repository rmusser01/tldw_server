"""TASK-15671: migration 109 adds the account-lock columns; the users write guard lets them through."""

import sqlite3
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ import migrations
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.migrations import migration_109_add_account_lock_columns
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user


def _db(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "users.db")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL)")
    conn.execute("CREATE TABLE api_keys (id INTEGER PRIMARY KEY, user_id INTEGER)")
    return conn


def _names(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_migration_109_adds_the_lock_columns(tmp_path) -> None:
    db = _db(tmp_path)
    migration_109_add_account_lock_columns(db)
    assert {"security_locked_at", "credentials_valid_after"} <= set(_names(db, "users"))
    assert "suspended_at" in _names(db, "api_keys")


def test_migration_109_is_idempotent(tmp_path) -> None:
    db = _db(tmp_path)
    migration_109_add_account_lock_columns(db)
    migration_109_add_account_lock_columns(db)
    assert _names(db, "users").count("security_locked_at") == 1
    assert _names(db, "api_keys").count("suspended_at") == 1


def test_migration_109_skips_missing_tables(tmp_path) -> None:
    """Synthetic databases migrate without a users or api_keys table (cf. migrations 100, 107 and 108)."""
    with sqlite3.connect(tmp_path / "synthetic.db") as conn:
        migration_109_add_account_lock_columns(conn)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not tables & {"users", "api_keys"}


def test_migration_109_is_registered_last() -> None:
    assert migrations.get_authnz_migrations()[-1].version == 109


@pytest.fixture
async def owner_db(tmp_path, monkeypatch, fresh_services):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("JWT_SECRET_KEY", "synthetic-signing-key-for-regression-tests-only")
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    reset_settings()
    pool = await get_db_pool()
    user_id = await create_authnz_test_user(pool, username="owner", email="owner@example.com")
    try:
        yield SimpleNamespace(pool=pool, user_id=user_id)
    finally:
        reset_settings()


@pytest.mark.asyncio
async def test_raw_update_of_lock_columns_passes_the_users_write_guard(owner_db) -> None:
    await owner_db.pool.execute(
        "UPDATE users SET security_locked_at = CURRENT_TIMESTAMP, credentials_valid_after = CURRENT_TIMESTAMP "
        "WHERE id = ?",
        owner_db.user_id,
    )
    row = dict(
        await owner_db.pool.fetchone(
            "SELECT security_locked_at, credentials_valid_after FROM users WHERE id = ?", owner_db.user_id
        )
    )
    assert row["security_locked_at"] is not None
    assert row["credentials_valid_after"] is not None
