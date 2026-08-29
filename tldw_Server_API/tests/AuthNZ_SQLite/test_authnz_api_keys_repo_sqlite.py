import json
import uuid

import pytest

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


@pytest.mark.asyncio
async def test_authnz_api_keys_repo_fetch_key_limits_sqlite(isolated_test_environment):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    _client, _db_name = isolated_test_environment
    pool = await get_db_pool()

    # Ensure core AuthNZ tables exist via UsersDB and APIKeyManager so we
    # reuse centralized schema logic.
    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="vk_repo_user",
        email="vk_repo_user@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    user_id = int(created_user["id"])

    mgr = APIKeyManager(pool)
    await mgr.initialize()
    vk = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk_repo_key",
        budget_day_tokens=123,
        budget_month_usd=4.56,
    )
    key_id = int(vk["id"])

    repo = AuthnzApiKeysRepo(db_pool=pool)
    row = await repo.fetch_key_limits(key_id)

    assert row is not None
    assert int(row["id"]) == key_id
    assert bool(row.get("is_virtual")) is True
    assert int(row.get("llm_budget_day_tokens") or 0) == 123
    assert pytest.approx(float(row.get("llm_budget_month_usd") or 0.0), rel=1e-6) == 4.56


@pytest.mark.asyncio
async def test_authnz_api_keys_repo_virtual_key_scope_alignment_sqlite(isolated_test_environment):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    _client, _db_name = isolated_test_environment
    pool = await get_db_pool()

    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="vk_scope_user",
        email="vk_scope_user@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    user_id = int(created_user["id"])

    mgr = APIKeyManager(pool)
    await mgr.initialize()
    created = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-scope-check",
        expires_in_days=1,
    )
    key_id = int(created["id"])

    row = await pool.fetchrow(
        "SELECT scope, status, expires_at, is_virtual, parent_key_id FROM api_keys WHERE id = ?",
        key_id,
    )
    assert row is not None
    assert row["scope"] == "read"
    assert row["status"] == "active"
    assert row["expires_at"] is not None
    assert row["parent_key_id"] is None
    assert bool(row["is_virtual"]) is True


@pytest.mark.asyncio
async def test_api_key_scope_list_serializes_and_normalizes_sqlite(isolated_test_environment):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager, normalize_scope

    _client, _db_name = isolated_test_environment
    pool = await get_db_pool()

    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="scope_list_user",
        email="scope_list_user@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    user_id = int(created_user["id"])

    mgr = APIKeyManager(pool)
    await mgr.initialize()

    scopes = ["read", "write"]
    created = await mgr.create_api_key(
        user_id=user_id,
        name="scope-list-key",
        scope=scopes,
    )
    key_id = int(created["id"])

    row = await pool.fetchrow(
        "SELECT scope FROM api_keys WHERE id = ?",
        key_id,
    )
    assert row is not None
    assert row["scope"] == json.dumps(scopes)

    normalized = normalize_scope(row["scope"])
    assert normalized == {"read", "write"}
    for scope in scopes:
        assert mgr._has_scope(row["scope"], scope) is True


@pytest.mark.asyncio
async def test_authnz_api_keys_repo_rotation_and_revoke_sqlite(isolated_test_environment):
    """AuthnzApiKeysRepo mark_rotated / revoke_api_key_for_user should work on SQLite."""
    from datetime import datetime

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager, APIKeyStatus

    _client, _db_name = isolated_test_environment
    pool = await get_db_pool()

    # Ensure user and tables exist
    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="repo_rot_user",
        email="repo_rot_user@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    user_id = int(created_user["id"])

    mgr = APIKeyManager(pool)
    await mgr.initialize()

    # Create two API keys for rotation tests
    first = await mgr.create_api_key(user_id=user_id, name="first")
    second = await mgr.create_api_key(user_id=user_id, name="second")
    first_id = int(first["id"])
    second_id = int(second["id"])

    repo = AuthnzApiKeysRepo(db_pool=pool)

    # mark_rotated: first -> second
    await repo.mark_rotated(
        old_key_id=first_id,
        new_key_id=second_id,
        rotated_status=APIKeyStatus.ROTATED.value,
        reason="Rotation test",
        revoked_at=datetime.utcnow(),
    )

    row_first = await pool.fetchrow(
        "SELECT status, rotated_to, revoke_reason FROM api_keys WHERE id = ?",
        first_id,
    )
    row_second = await pool.fetchrow(
        "SELECT rotated_from FROM api_keys WHERE id = ?",
        second_id,
    )
    assert row_first is not None
    assert row_second is not None
    assert row_first["status"] == APIKeyStatus.ROTATED.value
    assert int(row_first["rotated_to"]) == second_id
    assert row_first["revoke_reason"] == "Rotation test"
    assert int(row_second["rotated_from"]) == first_id

    # revoke_api_key_for_user: revoke second key
    revoked = await repo.revoke_api_key_for_user(
        key_id=second_id,
        user_id=user_id,
        revoked_status=APIKeyStatus.REVOKED.value,
        active_status=APIKeyStatus.ACTIVE.value,
        reason="Unit revoke",
        revoked_at=datetime.utcnow(),
    )
    assert revoked is True

    row_second_after = await pool.fetchrow(
        "SELECT status, revoked_by, revoke_reason FROM api_keys WHERE id = ?",
        second_id,
    )
    assert row_second_after is not None
    assert row_second_after["status"] == APIKeyStatus.REVOKED.value
    assert int(row_second_after["revoked_by"]) == user_id
    assert row_second_after["revoke_reason"] == "Unit revoke"


@pytest.mark.asyncio
async def test_authnz_api_keys_repo_usage_and_audit_sqlite(isolated_test_environment):
    """AuthnzApiKeysRepo.increment_usage and insert_audit_log work on SQLite."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    _client, _db_name = isolated_test_environment
    pool = await get_db_pool()

    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="vk_repo_usage_user",
        email="vk_repo_usage_user@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    user_id = int(created_user["id"])

    mgr = APIKeyManager(pool)
    await mgr.initialize()
    rec = await mgr.create_api_key(user_id=user_id, name="usage-key")
    key_id = int(rec["id"])

    repo = AuthnzApiKeysRepo(db_pool=pool)

    # usage increment
    before = await pool.fetchrow(
        "SELECT usage_count, last_used_at, last_used_ip FROM api_keys WHERE id = ?",
        key_id,
    )
    before_count = int(before["usage_count"] or 0) if before is not None else 0

    await repo.increment_usage(key_id=key_id, ip_address="127.0.0.1")

    after = await pool.fetchrow(
        "SELECT usage_count, last_used_at, last_used_ip FROM api_keys WHERE id = ?",
        key_id,
    )
    assert after is not None
    assert int(after["usage_count"] or 0) == before_count + 1
    assert after["last_used_ip"] == "127.0.0.1"
    assert after["last_used_at"] is not None

    # audit log insert
    before_audit = await pool.fetchval("SELECT COUNT(*) FROM api_key_audit_log")
    await repo.insert_audit_log(
        key_id=key_id,
        action="unit_test",
        user_id=user_id,
        details={"foo": "bar"},
    )
    after_audit = await pool.fetchval("SELECT COUNT(*) FROM api_key_audit_log")
    assert int(after_audit or 0) == int(before_audit or 0) + 1

    last_row = await pool.fetchrow(
        "SELECT api_key_id, action, user_id, details FROM api_key_audit_log ORDER BY id DESC LIMIT 1"
    )
    assert last_row is not None
    assert int(last_row["api_key_id"]) == key_id
    assert last_row["action"] == "unit_test"
    assert int(last_row["user_id"]) == user_id
