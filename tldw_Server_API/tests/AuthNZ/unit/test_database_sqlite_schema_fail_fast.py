import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError
from tldw_Server_API.app.core.AuthNZ.migrations import get_authnz_migrations
from tldw_Server_API.app.core.AuthNZ.settings import Settings


pytestmark = pytest.mark.unit


def _seed_current_but_drifted_authnz_sqlite_db(db_path) -> None:  # noqa: ANN001
    latest_version = max(migration.version for migration in get_authnz_migrations())
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE api_key_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO schema_migrations (version, name, applied_at)
            VALUES (?, ?, datetime('now'))
            """,
            (latest_version, "current"),
        )
        conn.commit()


def _seed_current_sqlite_db_with_legacy_scope_default(db_path) -> None:  # noqa: ANN001
    latest_version = max(migration.version for migration in get_authnz_migrations())
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                key_id TEXT,
                key_prefix TEXT,
                name TEXT,
                description TEXT,
                scope TEXT DEFAULT 'read',
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                last_used_at TIMESTAMP,
                last_used_ip TEXT,
                usage_count INTEGER DEFAULT 0,
                rate_limit INTEGER,
                allowed_ips TEXT,
                metadata TEXT,
                rotated_from INTEGER,
                rotated_to INTEGER,
                revoked_at TIMESTAMP,
                revoked_by INTEGER,
                revoke_reason TEXT,
                is_virtual INTEGER DEFAULT 0,
                parent_key_id INTEGER,
                org_id INTEGER,
                team_id INTEGER,
                llm_budget_day_tokens INTEGER,
                llm_budget_month_tokens INTEGER,
                llm_budget_day_usd REAL,
                llm_budget_month_usd REAL,
                llm_allowed_endpoints TEXT,
                llm_allowed_providers TEXT,
                llm_allowed_models TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE api_key_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                user_id INTEGER,
                ip_address TEXT,
                user_agent TEXT,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO schema_migrations (version, name, applied_at)
            VALUES (?, ?, datetime('now'))
            """,
            (latest_version, "current"),
        )
        conn.commit()


@pytest.mark.asyncio
async def test_database_pool_raises_when_sqlite_harmonization_fails_in_strict_mode(monkeypatch, tmp_path):
    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{tmp_path / 'strict.db'}",
        JWT_SECRET_KEY="x" * 64,
    )
    pool = DatabasePool(settings)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.ensure_authnz_tables",
        lambda _path: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.is_test_mode", lambda: False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.is_explicit_pytest_runtime",
        lambda: False,
    )

    with pytest.raises(DatabaseError, match="boom"):
        await pool.initialize()


@pytest.mark.asyncio
async def test_database_pool_allows_sqlite_harmonization_failure_in_test_mode(monkeypatch, tmp_path):
    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{tmp_path / 'test-mode.db'}",
        JWT_SECRET_KEY="x" * 64,
    )
    pool = DatabasePool(settings)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.ensure_authnz_tables",
        lambda _path: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.is_test_mode", lambda: True)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.is_explicit_pytest_runtime",
        lambda: False,
    )

    await pool.initialize()
    await pool.close()


@pytest.mark.asyncio
async def test_database_pool_raises_when_current_sqlite_schema_is_drifted_in_strict_mode(monkeypatch, tmp_path):
    db_path = tmp_path / "drifted.db"
    _seed_current_but_drifted_authnz_sqlite_db(db_path)

    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{db_path}",
        JWT_SECRET_KEY="x" * 64,
    )
    pool = DatabasePool(settings)

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.is_test_mode", lambda: False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.is_explicit_pytest_runtime",
        lambda: False,
    )

    with pytest.raises(DatabaseError, match="missing required columns"):
        await pool.initialize()


@pytest.mark.asyncio
async def test_database_pool_raises_when_current_sqlite_schema_keeps_legacy_scope_default(monkeypatch, tmp_path):
    db_path = tmp_path / "legacy-scope-default.db"
    _seed_current_sqlite_db_with_legacy_scope_default(db_path)

    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{db_path}",
        JWT_SECRET_KEY="x" * 64,
    )
    pool = DatabasePool(settings)

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.is_test_mode", lambda: False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.is_explicit_pytest_runtime",
        lambda: False,
    )

    with pytest.raises(DatabaseError, match="must not define a default"):
        await pool.initialize()
