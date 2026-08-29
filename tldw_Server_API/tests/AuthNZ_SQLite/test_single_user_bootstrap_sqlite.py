"""
SQLite-specific tests for single-user bootstrap flow.

Validates that bootstrap_single_user_profile correctly creates the admin user
and primary API key in SQLite, and that the bootstrap is idempotent.
"""

from pathlib import Path

import pytest
from loguru import logger


@pytest.mark.asyncio
async def test_single_user_bootstrap_creates_admin_user_and_primary_key(tmp_path, monkeypatch):
    # Configure single-user SQLite AuthNZ
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test_single_user_primary_key_123")

    # Reset AuthNZ singletons and ensure core tables exist
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Run the single-user bootstrap helper twice to assert idempotency
    from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile

    ok_first = await bootstrap_single_user_profile()
    ok_second = await bootstrap_single_user_profile()
    assert ok_first is True
    assert ok_second is True

    # Verify the single-user admin row exists with the fixed ID
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    settings = get_settings()
    single_user_id = settings.SINGLE_USER_FIXED_ID

    user_rows = await pool.fetch(
        "SELECT id, username, role, is_active, is_verified FROM users WHERE id = ?",
        single_user_id,
    )
    assert len(user_rows) == 1
    user = user_rows[0]
    assert int(user["id"]) == single_user_id
    assert user["username"] == "single_user"
    assert user["role"] == "admin"
    assert int(user["is_active"]) == 1
    assert int(user["is_verified"]) == 1

    # Verify a non-virtual primary API key row exists for this user
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    manager = APIKeyManager()
    await manager.initialize()
    key_hash = manager.hash_api_key(settings.SINGLE_USER_API_KEY)

    rows = await pool.fetch(
        "SELECT user_id, key_hash, scope, status, is_virtual FROM api_keys WHERE key_hash = ?",
        key_hash,
    )
    assert len(rows) == 1
    row = rows[0]
    assert int(row["user_id"]) == single_user_id
    assert row["key_hash"] == key_hash
    assert row["scope"] == "admin"
    assert row["status"] == "active"
    # SQLite uses 0/1 for booleans
    assert int(row["is_virtual"]) == 0


@pytest.mark.asyncio
async def test_single_user_bootstrap_is_idempotent_for_new_format_api_key(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.api_key_crypto import format_api_key

    db_path = tmp_path / "users_new_format.db"
    primary_key = format_api_key("abcdef123456", "new-format-secret")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("SINGLE_USER_API_KEY", primary_key)
    monkeypatch.delenv("SINGLE_USER_TEST_API_KEY", raising=False)

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings, get_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile

    ok_first = await bootstrap_single_user_profile()
    ok_second = await bootstrap_single_user_profile()
    assert ok_first is True
    assert ok_second is True

    settings = get_settings()
    rows = await pool.fetch(
        """
        SELECT user_id, key_id, scope, status, is_virtual
        FROM api_keys
        WHERE user_id = ? AND status = 'active' AND COALESCE(is_virtual, 0) = 0
        """,
        settings.SINGLE_USER_FIXED_ID,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["key_id"] == "abcdef123456"
    assert row["scope"] == "admin"


@pytest.mark.asyncio
async def test_single_user_bootstrap_reuses_preseeded_primary_key(tmp_path, monkeypatch):
    # Configure single-user SQLite AuthNZ with a deterministic key
    db_path = tmp_path / "users_preseeded.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test_single_user_preseeded_key_123")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings, get_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    settings = get_settings()
    single_user_id = settings.SINGLE_USER_FIXED_ID

    # Ensure API key tables exist and compute the hash for the configured key
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    manager = APIKeyManager()
    await manager.initialize()
    key_value = settings.SINGLE_USER_API_KEY
    key_hash = manager.hash_api_key(key_value)
    key_prefix = (key_value[:10] + "...") if len(key_value) > 10 else key_value

    # Pre-seed a primary key row for SINGLE_USER_API_KEY to simulate an existing deployment
    async with pool.transaction() as conn:  # type: ignore[attr-defined]
        await conn.execute(
            """
            INSERT OR IGNORE INTO users (id, username, email, password_hash, is_active, is_verified, role)
            VALUES (?, ?, ?, ?, 1, 1, 'admin')
            """,
            (single_user_id, "single_user", "single_user@example.local", ""),
        )
        await conn.execute(
            """
            INSERT INTO api_keys (
                user_id, key_hash, key_prefix, name, description,
                scope, status, is_virtual
            ) VALUES (?, ?, ?, ?, ?, ?, 'active', 0)
            """,
            (
                single_user_id,
                key_hash,
                key_prefix,
                "legacy primary key",
                "Pre-seeded primary API key row",
                "read",
            ),
        )
        try:
            await conn.commit()  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError) as e:
            # Some adapters commit implicitly on context exit
            logger.debug(f"Explicit commit skipped (adapter may auto-commit): {e}")

    before_rows = await pool.fetch(
        "SELECT id, user_id, key_hash, scope, status, is_virtual FROM api_keys WHERE key_hash = ?",
        key_hash,
    )
    assert len(before_rows) == 1
    existing = before_rows[0]
    existing_id = existing["id"]
    assert int(existing["user_id"]) == single_user_id
    assert existing["scope"] == "read"
    assert existing["status"] == "active"

    # Run bootstrap; it should upsert the existing row rather than creating a new one
    from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile

    ok = await bootstrap_single_user_profile()
    assert ok is True

    after_rows = await pool.fetch(
        "SELECT id, user_id, key_hash, scope, status, is_virtual FROM api_keys WHERE key_hash = ?",
        key_hash,
    )
    assert len(after_rows) == 1
    row = after_rows[0]
    # The same row is reused and upgraded to admin scope
    assert row["id"] == existing_id
    assert int(row["user_id"]) == single_user_id
    assert row["scope"] == "admin"
    assert row["status"] == "active"
    assert int(row["is_virtual"]) == 0


@pytest.mark.asyncio
async def test_single_user_bootstrap_fails_with_extra_active_user_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "users_conflict.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test_single_user_conflict_key_123")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Pre-seed an additional active user to trigger bootstrap failure.
    await pool.execute(
        """
        INSERT INTO users (id, username, email, password_hash, is_active, is_verified, role)
        VALUES (?, ?, ?, ?, 1, 1, 'user')
        """,
        999,
        "extra_user",
        "extra@example.local",
        "",
    )

    from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile

    ok = await bootstrap_single_user_profile()
    assert ok is False


@pytest.mark.asyncio
async def test_single_user_bootstrap_fails_with_multiple_primary_keys_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "users_multiple_keys.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test_single_user_multi_key_123")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings, get_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    settings = get_settings()
    single_user_id = settings.SINGLE_USER_FIXED_ID

    # Ensure the single-user row exists for FK constraints.
    await pool.execute(
        """
        INSERT OR IGNORE INTO users (id, username, email, password_hash, is_active, is_verified, role)
        VALUES (?, ?, ?, ?, 1, 1, 'admin')
        """,
        single_user_id,
        "single_user",
        "single_user@example.local",
        "",
    )

    manager = APIKeyManager()
    await manager.initialize()

    primary_value = settings.SINGLE_USER_API_KEY
    primary_hash = manager.hash_api_key(primary_value)
    primary_prefix = (primary_value[:10] + "...") if len(primary_value) > 10 else primary_value

    extra_value = "extra_single_user_key_456"
    extra_hash = manager.hash_api_key(extra_value)
    extra_prefix = (extra_value[:10] + "...") if len(extra_value) > 10 else extra_value

    await pool.execute(
        """
        INSERT INTO api_keys (
            user_id, key_hash, key_prefix, name, description,
            scope, status, is_virtual
        ) VALUES (?, ?, ?, ?, ?, ?, 'active', 0)
        """,
        single_user_id,
        primary_hash,
        primary_prefix,
        "primary key",
        "Primary API key",
        "admin",
    )
    await pool.execute(
        """
        INSERT INTO api_keys (
            user_id, key_hash, key_prefix, name, description,
            scope, status, is_virtual
        ) VALUES (?, ?, ?, ?, ?, ?, 'active', 0)
        """,
        single_user_id,
        extra_hash,
        extra_prefix,
        "extra key",
        "Extra API key",
        "read",
    )

    from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile

    ok = await bootstrap_single_user_profile()
    assert ok is False
