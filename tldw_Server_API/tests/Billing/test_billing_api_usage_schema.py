"""UAT292: billing reads the real AuthNZ daily usage schema on both engines."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.Billing.enforcement import BillingEnforcer

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture(params=["sqlite", "postgres"])
async def usage_pool(request, tmp_path, monkeypatch):
    """Use canonical initialization; PostgreSQL is provisioned by the official fixture."""
    if request.param == "postgres":
        pg = request.getfixturevalue("pg_temp_db")
        url = (
            f"postgresql://{quote(str(pg['user']), safe='')}:"
            f"{quote(str(pg['password']), safe='')}@{pg['host']}:{pg['port']}/{pg['database']}"
        )
    else:
        url = f"sqlite:///{tmp_path / 'users.db'}"
    pool = DatabasePool(Settings(
        AUTH_MODE="multi_user", DATABASE_URL=url,
        JWT_SECRET_KEY="uat292-test-secret-with-at-least-32-characters",
        DATABASE_POOL_MIN_SIZE=1, DATABASE_POOL_MAX_SIZE=5,
    ))
    async def get_pool():
        return pool

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.get_db_pool", get_pool)
    try:
        await pool.initialize()
        if pool.pool is not None:
            from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                ensure_api_keys_tables_pg,
                ensure_authnz_core_tables_pg,
                ensure_usage_tables_pg,
            )

            # Production bootstrap order: core -> API keys -> usage foreign keys.
            assert await ensure_authnz_core_tables_pg(pool)
            assert await ensure_api_keys_tables_pg(pool)
            assert await ensure_usage_tables_pg(pool)
        else:
            from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables

            ensure_authnz_tables(Path(pool.db_path))
        yield pool
    finally:
        await pool.close()


async def _user(pool: DatabasePool) -> int:
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    identifier = uuid4().hex
    users = UsersDB(pool)
    await users.initialize(ensure_schema=False)
    user = await users.create_user(
        username=f"uat292-{identifier}", email=f"{identifier}@example.test",
        password_hash="unused-test-hash",
    )
    return int(user["id"])


async def _org(pool: DatabasePool) -> int:
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo

    identifier = uuid4().hex
    org = await AuthnzOrgsTeamsRepo(pool).create_organization(name=identifier, slug=identifier)
    return int(org["id"])


async def _membership(pool: DatabasePool, user: int, org: int, days_ago: int) -> None:
    added = datetime(2026, 1, 10, tzinfo=timezone.utc) - timedelta(days=days_ago)
    value = added if pool.pool is not None else added.isoformat()
    await pool.execute(
        "INSERT INTO org_members (user_id, org_id, added_at) VALUES (?, ?, ?)", user, org, value,
    )


async def _daily(pool: DatabasePool, user: int, requests: int, days_offset: int = 0) -> None:
    day = datetime.now(timezone.utc).date() + timedelta(days=days_offset)
    value = day if pool.pool is not None else day.isoformat()
    await pool.execute(
        "INSERT INTO usage_daily (user_id, day, requests) VALUES (?, ?, ?)", user, value, requests,
    )


async def test_current_api_usage_counts_real_rollup_once_per_primary_org(usage_pool):
    """Wrong schema/joins must not erase usage or charge another org's users."""
    pool = usage_pool
    org_a, org_b, empty_org = [await _org(pool) for _ in range(3)]
    alice, shared, bob, orphan = [await _user(pool) for _ in range(4)]
    await _membership(pool, alice, org_a, 2)
    await _membership(pool, shared, org_b, 1)
    await _membership(pool, shared, org_a, 2)
    await _membership(pool, bob, org_b, 2)
    repo = AuthnzUsageRepo(pool)
    for user, count in [(alice, 2), (shared, 3), (bob, 7), (orphan, 11)]:
        for _ in range(count):
            await repo.insert_usage_log(
                user_id=user, key_id=None, endpoint="GET:/uat292", status=200,
                latency_ms=1, bytes_out=0, bytes_in=0, meta="{}", request_id=uuid4().hex,
            )
    await repo.aggregate_usage_daily_for_day()
    await _daily(pool, alice, 101, -1)
    await _daily(pool, alice, 103, 1)
    enforcer = BillingEnforcer(cache_ttl=0)
    assert [await enforcer._get_api_calls_today(org) for org in (org_a, org_b, empty_org)] == [5, 7, 0]


@pytest.mark.parametrize("earlier_org", ["higher", "tie"])
async def test_api_usage_primary_org_uses_membership_time_then_id(usage_pool, earlier_org):
    """A later membership or a tied higher ID must not duplicate a user's count."""
    pool = usage_pool
    lower, higher = [await _org(pool) for _ in range(2)]
    user = await _user(pool)
    await _membership(pool, user, higher, 2)
    await _membership(pool, user, lower, 1 if earlier_org == "higher" else 2)
    await _daily(pool, user, 13)
    expected = [0, 13] if earlier_org == "higher" else [13, 0]
    enforcer = BillingEnforcer()
    assert [await enforcer._get_api_calls_today(org) for org in (lower, higher)] == expected


@pytest.mark.parametrize("failure_mode", ["open", "closed"])
async def test_unavailable_daily_usage_preserves_failure_policy(usage_pool, monkeypatch, failure_mode):
    """An unavailable source follows the configured policy and never widens the query."""
    pool = usage_pool
    org = await _org(pool)
    monkeypatch.setenv("BILLING_ENFORCEMENT_FAILURE_MODE", failure_mode)
    # Administrative fault injection is limited to this disposable fixture DB.
    # Managed AuthNZ connections intentionally reject schema-destructive writes.
    if pool.pool is not None:
        import asyncpg

        conn = await asyncpg.connect(pool.settings.DATABASE_URL)
        try:
            await conn.execute("ALTER TABLE usage_daily RENAME TO unavailable_usage_daily")
        finally:
            await conn.close()
    else:
        import sqlite3

        with sqlite3.connect(pool.db_path) as conn:
            conn.execute("ALTER TABLE usage_daily RENAME TO unavailable_usage_daily")
    enforcer = BillingEnforcer()
    if failure_mode == "closed":
        from tldw_Server_API.app.core.Billing.enforcement import _BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS

        with pytest.raises(_BILLING_ENFORCEMENT_NONCRITICAL_EXCEPTIONS):
            await enforcer._get_api_calls_today(org)
        assert (await enforcer.get_org_usage(org)).api_calls_today == 2_147_483_647
    else:
        assert await enforcer._get_api_calls_today(org) == 0
    assert await pool.fetchval("SELECT 1") == 1


@pytest.mark.parametrize("usage_pool", ["sqlite"], indirect=True)
@pytest.mark.parametrize("all_undated", [False, True])
async def test_legacy_undated_memberships_match_existing_billing_attribution(usage_pool, all_undated):
    """Legacy NULL timestamps must not outrank dated memberships or invent a primary org."""
    import sqlite3

    from tldw_Server_API.app.core.AuthNZ.migrations import migration_016_create_orgs_teams

    pool = usage_pool
    # Reconstruct the canonical legacy table; fresh bootstrap enforces NOT NULL.
    with sqlite3.connect(pool.db_path) as conn:
        conn.execute("DROP TABLE org_members")
        migration_016_create_orgs_teams(conn)
    lower, higher = [await _org(pool) for _ in range(2)]
    user = await _user(pool)
    await _membership(pool, user, lower, 2)
    await _membership(pool, user, higher, 1)
    await _daily(pool, user, 13)
    await pool.execute("UPDATE org_members SET added_at = NULL WHERE org_id = ?", lower)
    if all_undated:
        await pool.execute("UPDATE org_members SET added_at = NULL WHERE org_id = ?", higher)
    enforcer = BillingEnforcer()
    expected = [0, 0] if all_undated else [0, 13]
    assert [await enforcer._get_api_calls_today(org) for org in (lower, higher)] == expected
