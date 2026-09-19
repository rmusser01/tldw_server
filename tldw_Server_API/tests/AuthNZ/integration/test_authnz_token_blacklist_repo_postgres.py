from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Literal

import asyncpg
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.repos.token_blacklist_repo import (
    AuthnzTokenBlacklistRepo,
)
from tldw_Server_API.app.core.AuthNZ.token_blacklist import TokenBlacklist

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def blacklist_schema_admin(
    isolated_test_environment: tuple[TestClient, str],
) -> AsyncIterator[asyncpg.Connection]:
    """Own fault-setup access to the official fixture's isolated database.

    Application pool connections intentionally reject destructive DDL. This
    administrative connection cannot outlive the fixture that provisions its DB.
    """
    connection = await asyncpg.connect(os.environ["TEST_DATABASE_URL"])
    try:
        assert await connection.fetchval("SELECT current_database()") == isolated_test_environment[1]
        yield connection
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_object", ["table", "index"])
@pytest.mark.parametrize("first_completion", ["commit", "cancel"])
async def test_concurrent_blacklist_bootstrap_preserves_revocation(
    isolated_test_environment: tuple[TestClient, str],
    blacklist_schema_admin: asyncpg.Connection,
    missing_object: Literal["table", "index"],
    first_completion: Literal["commit", "cancel"],
) -> None:
    """Overlapping real DDL must commit safely, including partial index repair."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    _client, database_name = isolated_test_environment
    pool = await get_db_pool()
    assert await pool.fetchval("SELECT current_database()") == database_name
    repo = AuthnzTokenBlacklistRepo(pool)
    # Deliberate schema damage uses the fixture-owned administrative connection.
    await blacklist_schema_admin.execute("DROP TABLE IF EXISTS token_blacklist")
    if missing_object == "index":
        await repo.ensure_schema()
        await blacklist_schema_admin.execute("DROP INDEX idx_blacklist_jti")

    created = asyncio.Event()
    release = asyncio.Event()
    second_acquired = asyncio.Event()
    second_pid = None
    held_statement = (
        "CREATE TABLE IF NOT EXISTS token_blacklist"
        if missing_object == "table"
        else "CREATE INDEX IF NOT EXISTS idx_blacklist_jti"
    )

    class HoldFirstDDL:
        """Hold the first DDL inside its real transaction until the competing lock is observed."""

        def __init__(self, connection: asyncpg.Connection) -> None:
            """Borrow a fixture-managed connection without owning its cleanup."""
            self.connection = connection

        async def execute(self, query: str, *args: Any) -> str:
            """Execute real SQL, then hold the chosen DDL; propagate errors and cancellation."""
            result = await self.connection.execute(query, *args)
            if query.strip().startswith(held_statement):
                created.set()
                await release.wait()
            return result

    @asynccontextmanager
    async def first_transaction() -> AsyncIterator[HoldFirstDDL]:
        """Yield the gated executor; the fixture pool commits or rolls back on exit."""
        async with pool.transaction() as connection:
            yield HoldFirstDDL(connection)

    @asynccontextmanager
    async def second_transaction() -> AsyncIterator[asyncpg.Connection]:
        """Expose the contender's PID before yielding its managed transaction connection."""
        nonlocal second_pid
        async with pool.transaction() as connection:
            second_pid = connection.get_server_pid()
            second_acquired.set()
            yield connection

    first_repo = AuthnzTokenBlacklistRepo(
        SimpleNamespace(pool=pool.pool, transaction=first_transaction)
    )
    second_repo = AuthnzTokenBlacklistRepo(
        SimpleNamespace(pool=pool.pool, transaction=second_transaction)
    )

    async def wait_for_database_overlap() -> None:
        """Wait for the real contender lock, bounded by the caller's five-second timeout."""
        while not await pool.fetchval(
            "SELECT wait_event_type = 'Lock' FROM pg_stat_activity WHERE pid = $1",
            second_pid,
        ):
            await asyncio.sleep(0.01)

    async with asyncio.TaskGroup() as tasks:
        first = tasks.create_task(first_repo.ensure_schema())
        try:
            await asyncio.wait_for(created.wait(), timeout=5)
            tasks.create_task(second_repo.ensure_schema())
            await asyncio.wait_for(second_acquired.wait(), timeout=5)
            await asyncio.wait_for(wait_for_database_overlap(), timeout=5)
            if first_completion == "cancel":
                first.cancel()
        finally:
            release.set()

    now = datetime.now(timezone.utc).replace(microsecond=0)
    await repo.insert_blacklisted_token(
        jti="concurrent-bootstrap-revoked",
        user_id=None,
        token_type="access",
        expires_at=now + timedelta(hours=1),
        reason="concurrency-regression",
        revoked_by=None,
        ip_address=None,
    )
    assert await repo.get_active_expiry_for_jti("concurrent-bootstrap-revoked", now)
    assert await repo.get_active_expiry_for_jti("not-revoked", now) is None
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'token_blacklist' "
        "AND indexname IN ('idx_blacklist_jti', 'idx_blacklist_expires', 'idx_blacklist_user')"
    ) == 3


@pytest.mark.asyncio
async def test_authnz_token_blacklist_repo_postgres(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    """AuthnzTokenBlacklistRepo helpers should work on Postgres."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    _client, database_name = isolated_test_environment
    pool = await get_db_pool()
    assert await pool.fetchval("SELECT current_database()") == database_name

    # Ensure token_blacklist table exists in this test's isolated Postgres DB.
    # Use the real TokenBlacklist service bootstrap against this pool.
    service = TokenBlacklist(db_pool=pool)
    await service.initialize()

    repo = AuthnzTokenBlacklistRepo(pool)

    now = datetime.now(timezone.utc).replace(microsecond=0)
    future = now + timedelta(hours=2)

    # Insert a blacklisted token
    await repo.insert_blacklisted_token(
        jti="pg-test-jti",
        user_id=None,
        token_type="refresh",
        expires_at=future,
        reason="integration-test",
        revoked_by=None,
        ip_address=None,
    )

    # Active expiry lookup should find it
    expiry = await repo.get_active_expiry_for_jti("pg-test-jti", now=now)
    assert expiry is not None

    # Global stats should see at least one token
    stats = await repo.get_blacklist_stats(now=now, user_id=None)
    assert stats["total"] >= 1
    assert stats["refresh_tokens"] >= 1

    # Cleanup with past cutoff should keep it
    deleted_none = await repo.cleanup_expired(now=now - timedelta(hours=1))
    assert deleted_none == 0

    # Cleanup with future cutoff should be able to delete all expired rows.
    cutoff = now + timedelta(days=1)
    # Count rows that will be considered expired by this cutoff (including our test row).
    cutoff_naive = cutoff.replace(tzinfo=None)
    before_expired = await pool.fetchval(
        "SELECT COUNT(*) FROM token_blacklist WHERE expires_at < $1",
        cutoff_naive,
    )
    assert before_expired >= 1

    deleted_some = await repo.cleanup_expired(now=cutoff)
    # The repo should report the same number of deletions as actually removed rows.
    assert deleted_some == before_expired

    after_expired = await pool.fetchval(
        "SELECT COUNT(*) FROM token_blacklist WHERE expires_at < $1",
        cutoff_naive,
    )
    assert after_expired == 0
