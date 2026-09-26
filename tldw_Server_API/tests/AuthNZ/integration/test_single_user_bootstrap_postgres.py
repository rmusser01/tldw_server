import os

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_single_user_bootstrap_creates_admin_user_and_primary_key_postgres(
    isolated_test_environment,
    monkeypatch,
):
    # Use the isolated Postgres-backed AuthNZ environment
    client, _db_name = isolated_test_environment
    assert isinstance(client, TestClient)

    # Switch to single-user profile while keeping the Postgres DATABASE_URL
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test_single_user_primary_key_pg_123")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings, get_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile

    # Ensure AuthNZ settings and pool pick up the single-user profile
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()

    # Run the single-user bootstrap helper twice to assert idempotency
    ok_first = await bootstrap_single_user_profile()
    ok_second = await bootstrap_single_user_profile()
    assert ok_first is True
    assert ok_second is True

    settings = get_settings()
    single_user_id = settings.SINGLE_USER_FIXED_ID

    # Verify the single-user admin row exists with the fixed ID
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
    # Postgres uses boolean; SQLite uses 0/1 - normalize via int(...)
    assert int(row["is_virtual"]) == 0


@pytest.mark.asyncio
async def test_single_user_bootstrap_reuses_preseeded_primary_key_postgres(
    isolated_test_environment,
    monkeypatch,
):
    # Use the isolated Postgres-backed AuthNZ environment
    client, _db_name = isolated_test_environment
    assert isinstance(client, TestClient)

    # Switch to single-user profile while keeping the Postgres DATABASE_URL
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test_single_user_preseeded_key_pg_123")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings, get_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    settings = get_settings()
    single_user_id = settings.SINGLE_USER_FIXED_ID

    # Ensure API key tables exist and compute the hash for the configured key
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    manager = APIKeyManager()
    await manager.initialize()
    key_value = settings.SINGLE_USER_API_KEY
    key_hash = manager.hash_api_key(key_value)
    key_prefix = (key_value[:10] + "...") if len(key_value) > 10 else key_value

    # Pre-seed a primary key row for SINGLE_USER_API_KEY to simulate an existing deployment.
    # Use DatabasePool helpers so placeholder handling stays backend-agnostic.
    await create_authnz_test_user(
        pool,
        user_id=single_user_id,
        username="single_user",
        email="single_user@example.local",
        password_hash="",
        role="admin",
        is_verified=True,
        ignore_conflict=True,
    )
    await pool.execute(
        """
        INSERT INTO api_keys (
            user_id, key_hash, key_prefix, name, description,
            scope, status, is_virtual
        ) VALUES (?, ?, ?, ?, ?, ?, 'active', FALSE)
        ON CONFLICT (key_hash) DO NOTHING
        """,
        single_user_id,
        key_hash,
        key_prefix,
        "legacy primary key",
        "Pre-seeded primary API key row",
        "read",
    )

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
async def test_single_user_bootstrap_fails_with_extra_active_user_postgres(
    isolated_test_environment,
    monkeypatch,
):
    client, _db_name = isolated_test_environment
    assert isinstance(client, TestClient)

    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test_single_user_conflict_key_pg_123")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()

    await create_authnz_test_user(
        pool,
        user_id=999,
        username="extra_user",
        email="extra@example.local",
        password_hash="",
        is_verified=True,
        ignore_conflict=True,
    )

    ok = await bootstrap_single_user_profile()
    assert ok is False


@pytest.mark.asyncio
async def test_single_user_bootstrap_fails_with_multiple_primary_keys_postgres(
    isolated_test_environment,
    monkeypatch,
):
    client, _db_name = isolated_test_environment
    assert isinstance(client, TestClient)

    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test_single_user_multi_key_pg_123")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings, get_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    settings = get_settings()
    single_user_id = settings.SINGLE_USER_FIXED_ID

    await create_authnz_test_user(
        pool,
        user_id=single_user_id,
        username="single_user",
        email="single_user@example.local",
        password_hash="",
        role="admin",
        is_verified=True,
        ignore_conflict=True,
    )

    manager = APIKeyManager()
    await manager.initialize()

    primary_value = settings.SINGLE_USER_API_KEY
    primary_hash = manager.hash_api_key(primary_value)
    primary_prefix = (primary_value[:10] + "...") if len(primary_value) > 10 else primary_value

    extra_value = "extra_single_user_key_pg_456"
    extra_hash = manager.hash_api_key(extra_value)
    extra_prefix = (extra_value[:10] + "...") if len(extra_value) > 10 else extra_value

    await pool.execute(
        """
        INSERT INTO api_keys (
            user_id, key_hash, key_prefix, name, description,
            scope, status, is_virtual
        ) VALUES (?, ?, ?, ?, ?, ?, 'active', FALSE)
        ON CONFLICT (key_hash) DO NOTHING
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
        ) VALUES (?, ?, ?, ?, ?, ?, 'active', FALSE)
        ON CONFLICT (key_hash) DO NOTHING
        """,
        single_user_id,
        extra_hash,
        extra_prefix,
        "extra key",
        "Extra API key",
        "read",
    )

    ok = await bootstrap_single_user_profile()
    assert ok is False
