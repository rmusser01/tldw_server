from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager, APIKeyStatus
from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_authnz_api_keys_repo_rotation_and_revoke_postgres(test_db_pool):
    """AuthnzApiKeysRepo mark_rotated / revoke_api_key_for_user should work on Postgres."""
    pool = test_db_pool

    # Seed a user row for FK
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users (
                uuid,
                username,
                email,
                password_hash,
                role,
                is_active,
                is_verified,
                storage_quota_mb,
                storage_used_mb,
                created_at
            )
            VALUES ($1, $2, $3, $4, $5, TRUE, TRUE, 5120, 0.0, $6)
            """,
            str(uuid.uuid4()),
            "pg_api_keys_repo_user",
            "pg_api_keys_repo_user@example.com",
            "x",
            "user",
            datetime.utcnow(),
        )

    user_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = $1",
        "pg_api_keys_repo_user",
    )
    assert user_id is not None
    user_id_int = int(user_id)

    mgr = APIKeyManager(pool)
    await mgr.initialize()

    first = await mgr.create_api_key(user_id=user_id_int, name="pg-first")
    second = await mgr.create_api_key(user_id=user_id_int, name="pg-second")
    first_id = int(first["id"])
    second_id = int(second["id"])

    repo = AuthnzApiKeysRepo(db_pool=pool)

    # mark_rotated: first -> second
    await repo.mark_rotated(
        old_key_id=first_id,
        new_key_id=second_id,
        rotated_status=APIKeyStatus.ROTATED.value,
        reason="PG rotation",
        revoked_at=datetime.now(timezone.utc),
    )

    row_first = await pool.fetchrow(
        "SELECT status, rotated_to, revoke_reason FROM api_keys WHERE id = $1",
        first_id,
    )
    row_second = await pool.fetchrow(
        "SELECT rotated_from FROM api_keys WHERE id = $1",
        second_id,
    )
    assert row_first is not None
    assert row_second is not None
    assert row_first["status"] == APIKeyStatus.ROTATED.value
    assert int(row_first["rotated_to"]) == second_id
    assert row_first["revoke_reason"] == "PG rotation"
    assert int(row_second["rotated_from"]) == first_id

    # revoke_api_key_for_user: revoke second key
    revoked = await repo.revoke_api_key_for_user(
        key_id=second_id,
        user_id=user_id_int,
        revoked_status=APIKeyStatus.REVOKED.value,
        active_status=APIKeyStatus.ACTIVE.value,
        reason="PG revoke",
        revoked_at=datetime.now(timezone.utc),
    )
    assert revoked is True

    row_second_after = await pool.fetchrow(
        """
        SELECT status, revoked_by, revoke_reason
        FROM api_keys
        WHERE id = $1
        """,
        second_id,
    )
    assert row_second_after is not None
    assert row_second_after["status"] == APIKeyStatus.REVOKED.value
    assert int(row_second_after["revoked_by"]) == user_id_int
    assert row_second_after["revoke_reason"] == "PG revoke"


@pytest.mark.asyncio
async def test_authnz_api_keys_repo_usage_and_audit_postgres(test_db_pool):
    """AuthnzApiKeysRepo.increment_usage and insert_audit_log work on Postgres."""
    pool = test_db_pool

    # Seed a user row for FK
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users (
                uuid,
                username,
                email,
                password_hash,
                role,
                is_active,
                is_verified,
                storage_quota_mb,
                storage_used_mb,
                created_at
            )
            VALUES ($1, $2, $3, $4, $5, TRUE, TRUE, 5120, 0.0, $6)
            """,
            str(uuid.uuid4()),
            "pg_api_keys_usage_user",
            "pg_api_keys_usage_user@example.com",
            "x",
            "user",
            datetime.utcnow(),
        )

    user_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = $1",
        "pg_api_keys_usage_user",
    )
    assert user_id is not None
    user_id_int = int(user_id)

    mgr = APIKeyManager(pool)
    await mgr.initialize()

    rec = await mgr.create_api_key(user_id=user_id_int, name="pg-usage-key")
    key_id = int(rec["id"])

    repo = AuthnzApiKeysRepo(db_pool=pool)

    # usage increment
    before = await pool.fetchrow(
        "SELECT usage_count, last_used_at, last_used_ip FROM api_keys WHERE id = $1",
        key_id,
    )
    before_count = int(before["usage_count"] or 0) if before is not None else 0

    await repo.increment_usage(key_id=key_id, ip_address="203.0.113.5")

    after = await pool.fetchrow(
        "SELECT usage_count, last_used_at, last_used_ip FROM api_keys WHERE id = $1",
        key_id,
    )
    assert after is not None
    assert int(after["usage_count"] or 0) == before_count + 1
    assert after["last_used_ip"] == "203.0.113.5"
    assert after["last_used_at"] is not None

    # audit log insert
    before_audit = await pool.fetchval("SELECT COUNT(*) FROM api_key_audit_log")
    await repo.insert_audit_log(
        key_id=key_id,
        action="unit_test_pg",
        user_id=user_id_int,
        details={"foo": "bar"},
    )
    after_audit = await pool.fetchval("SELECT COUNT(*) FROM api_key_audit_log")
    assert int(after_audit or 0) == int(before_audit or 0) + 1

    last_row = await pool.fetchrow(
        """
        SELECT api_key_id, action, user_id, details
        FROM api_key_audit_log
        ORDER BY id DESC
        LIMIT 1
        """
    )
    assert last_row is not None
    assert int(last_row["api_key_id"]) == key_id
    assert last_row["action"] == "unit_test_pg"
    assert int(last_row["user_id"]) == user_id_int


@pytest.mark.asyncio
async def test_create_virtual_key_row_persists_text_and_jsonb_lists_postgres(test_db_pool):
    """Virtual-key inserts must pass the guard for text and JSONB columns."""
    pool = test_db_pool
    username = "pg_virtual_key_repo_user"
    users_db = UsersDB(pool)
    await users_db.initialize()
    user = await users_db.create_user(
        username=username,
        email=f"{username}@example.com",
        password_hash=uuid.uuid4().hex,
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
    )
    user_id = int(user["id"])

    repo = AuthnzApiKeysRepo(db_pool=pool)

    async def _create_key(name: str) -> int:
        async with pool.transaction() as conn:
            return await repo.create_virtual_key_row(
                user_id=user_id,
                key_hash=f"hash-{name}",
                key_identifier=f"id-{name}",
                key_prefix="vk_test...",
                name=name,
                description="virtual-key persistence coverage",
                expires_at=None,
                org_id=11,
                team_id=12,
                scope="read",
                allowed_endpoints=["chat.completions"],
                allowed_providers=["openai"],
                allowed_models=["gpt-test"],
                budget_day_tokens=100,
                budget_month_tokens=200,
                budget_day_usd=1.5,
                budget_month_usd=2.5,
                parent_key_id=None,
                allowed_methods=["post"],
                allowed_paths=["/api/v1/chat/completions"],
                max_calls=3,
                max_runs=4,
                conn=conn,
            )

    async def _assert_stored_lists(key_id: int) -> None:
        row = await pool.fetchrow(
            """
            SELECT is_virtual, scope, llm_allowed_endpoints, llm_allowed_providers,
                   llm_allowed_models, metadata
            FROM api_keys WHERE id = $1
            """,
            key_id,
        )
        assert row is not None
        assert row["is_virtual"] is True
        assert row["scope"] == "read"
        assert json.loads(row["llm_allowed_endpoints"]) == ["chat.completions"]
        assert json.loads(row["llm_allowed_providers"]) == ["openai"]
        assert json.loads(row["llm_allowed_models"]) == ["gpt-test"]
        assert json.loads(row["metadata"]) == {
            "allowed_methods": ["POST"],
            "allowed_paths": ["/api/v1/chat/completions"],
            "max_calls": 3,
            "max_runs": 4,
        }

    text_key_id = await _create_key("pg-virtual-text")
    await _assert_stored_lists(text_key_id)

    try:
        for column in (
            "llm_allowed_endpoints",
            "llm_allowed_providers",
            "llm_allowed_models",
        ):
            await pool.execute(
                f"ALTER TABLE api_keys ALTER COLUMN {column} TYPE JSONB USING {column}::jsonb"
            )
        jsonb_key_id = await _create_key("pg-virtual-jsonb")
        await _assert_stored_lists(jsonb_key_id)
    finally:
        for column in (
            "llm_allowed_endpoints",
            "llm_allowed_providers",
            "llm_allowed_models",
        ):
            await pool.execute(
                f"ALTER TABLE api_keys ALTER COLUMN {column} TYPE TEXT USING {column}::text"
            )
