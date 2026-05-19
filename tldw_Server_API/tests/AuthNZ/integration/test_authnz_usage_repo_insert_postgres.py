from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_authnz_usage_repo_insert_usage_log_postgres(test_db_pool):
    """AuthnzUsageRepo.insert_usage_log should insert rows on Postgres with bytes_in."""
    from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo

    pool = test_db_pool
    repo = AuthnzUsageRepo(pool)

    # Seed a user and API key to satisfy Postgres FKs on usage_log
    async with pool.acquire() as conn:
        user_id = await conn.fetchval(
            """
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES (gen_random_uuid(), $1, $2, $3, $4, TRUE, TRUE, 5120, 0.0)
            RETURNING id
            """,
            "pg-usage-user",
            "pg-usage-user@example.com",
            "hashed",
            "user",
        )
        key_id = await conn.fetchval(
            """
            INSERT INTO api_keys (user_id, key_hash, key_prefix, scope, status)
            VALUES ($1, $2, $3, $4, 'active')
            RETURNING id
            """,
            int(user_id),
            "pg-usage-key-hash",
            "pg-usage-prefix",
            "read",
        )

    await repo.insert_usage_log(
        user_id=int(user_id),
        key_id=int(key_id),
        endpoint="POST:/usage-test",
        status=201,
        latency_ms=75,
        bytes_out=2048,
        bytes_in=1024,
        meta='{"ip": "10.0.0.1"}',
        request_id="req-postgres-insert",
    )

    row = await pool.fetchone(
        """
        SELECT user_id, key_id, endpoint, status, latency_ms,
               bytes, bytes_in, meta, request_id
        FROM usage_log
        WHERE request_id = $1
        """,
        "req-postgres-insert",
    )

    assert row is not None
    row = dict(row)
    assert int(row["user_id"]) == int(user_id)
    assert int(row["key_id"]) == int(key_id)
    assert row["endpoint"] == "POST:/usage-test"
    assert int(row["status"]) == 201
    assert int(row["latency_ms"]) == 75
    assert int(row["bytes"]) == 2048
    assert int(row["bytes_in"]) == 1024
    assert '"ip": "10.0.0.1"' in (row.get("meta") or "")
    assert row["request_id"] == "req-postgres-insert"
