"""Reservation parity using the canonical per-test PostgreSQL environment."""

import asyncio
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio

from tldw_Server_API.tests.AuthNZ_Unit.test_provider_usage_reservations_repo import (
    ReservationContract,
    api,
    reservation,
)

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def reservation_pool(isolated_test_environment):
    from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_usage_tables_pg
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    pool = DatabasePool()
    await pool.initialize()
    try:
        assert await ensure_usage_tables_pg(pool)
        users = UsersDB(pool)
        await users.initialize(ensure_schema=False)
        user = await users.create_user(
            username="reservation-user",
            email="reservation@example.com",
            password_hash="hash",
            role="user",
            is_active=True,
            is_verified=True,
        )
        assert int(user["id"]) == 1
        yield pool
    finally:
        await pool.close()


class TestPostgresReservations(ReservationContract):
    pass


@pytest.mark.asyncio
async def test_repeatable_read_default_cannot_admit_using_prelock_snapshot(reservation_pool, monkeypatch):
    """A deployment's stronger default must not resurrect a stale quota snapshot."""
    mod, pool = api(), reservation_pool
    repo = mod.ProviderUsageReservationsRepo(pool)
    await repo.outstanding(mod.BillingScope("user", 1))
    original_transaction = pool.transaction

    @asynccontextmanager
    async def repeatable_read_transaction():
        async with original_transaction() as conn:
            await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            yield conn

    monkeypatch.setattr(pool, "transaction", repeatable_read_transaction)
    snapshot_entered, release_snapshot = asyncio.Event(), asyncio.Event()

    async def first_snapshot(_conn):
        snapshot_entered.set()
        await release_snapshot.wait()
        return mod.ReservationQuotaSnapshot(0, 30, 0, None)

    async def second_snapshot(_conn):
        return mod.ReservationQuotaSnapshot(0, 30, 0, None)

    first = asyncio.create_task(repo.reserve(reservation(), snapshot_reader=first_snapshot))
    await asyncio.wait_for(snapshot_entered.wait(), 5)
    second = asyncio.create_task(repo.reserve(reservation(), snapshot_reader=second_snapshot))
    try:
        await asyncio.sleep(0.05)
    finally:
        release_snapshot.set()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert sum(isinstance(result, dict) for result in results) == 1
    errors = [result for result in results if isinstance(result, Exception)]
    assert len(errors) == 1 and errors[0].code == "quota_exceeded"
