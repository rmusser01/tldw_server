import asyncio
import os
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import ResourceDailyLedger, LedgerEntry
from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool


@pytest.mark.asyncio
async def test_daily_ledger_insert_and_idempotency(tmp_path, monkeypatch):
    # Point AuthNZ DB to a temporary SQLite file
    db_path = tmp_path / "users_test.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    await reset_db_pool()
    pool = await get_db_pool()

    ledger = ResourceDailyLedger(db_pool=pool)
    await ledger.initialize()

    now = datetime.now(timezone.utc)
    e = LedgerEntry(
        entity_scope="user",
        entity_value="u123",
        category="minutes",
        units=5,
        op_id="op-1",
        occurred_at=now,
    )
    inserted = await ledger.add(e)
    assert inserted is True

    # Idempotent on same op_id
    inserted2 = await ledger.add(e)
    assert inserted2 is False

    # Sum for the day matches units
    total = await ledger.total_for_day("user", "u123", "minutes")
    assert total == 5

    # Different op_id accumulates
    e2 = LedgerEntry(
        entity_scope="user",
        entity_value="u123",
        category="minutes",
        units=7,
        op_id="op-2",
        occurred_at=now,
    )
    await ledger.add(e2)
    total2 = await ledger.total_for_day("user", "u123", "minutes")
    assert total2 == 12

    # Remaining helper
    remaining0 = await ledger.remaining_for_day("user", "u123", "minutes", daily_cap=10)
    assert remaining0 == 0
    remaining8 = await ledger.remaining_for_day("user", "u123", "minutes", daily_cap=20)
    assert remaining8 == 8

    # Peek range across two days
    # Add an entry for yesterday
    from datetime import timedelta
    yday = now - timedelta(days=1)
    e3 = LedgerEntry(
        entity_scope="user",
        entity_value="u123",
        category="minutes",
        units=3,
        op_id="op-3",
        occurred_at=yday,
    )
    await ledger.add(e3)
    start_day = yday.astimezone(timezone.utc).strftime("%Y-%m-%d")
    end_day = now.astimezone(timezone.utc).strftime("%Y-%m-%d")
    peek = await ledger.peek_range("user", "u123", "minutes", start_day, end_day)
    assert isinstance(peek, dict)
    assert peek.get("total") == 15  # 3 (yday) + 12 (today)
    days = {d["day_utc"]: d["units"] for d in peek.get("days", [])}
    assert days.get(start_day) == 3
    assert days.get(end_day) == 12


@pytest.mark.asyncio
async def test_daily_ledger_consume_if_within_cap_serializes_concurrent_writers(tmp_path, monkeypatch):
    db_path = tmp_path / "users_daily_cap_atomic.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    await reset_db_pool()
    pool = await get_db_pool()

    ledger = ResourceDailyLedger(db_pool=pool)
    await ledger.initialize()
    now = datetime.now(timezone.utc)

    async def consume(op_id: str):
        return await ledger.consume_if_within_cap(
            LedgerEntry(
                entity_scope="user",
                entity_value="u-atomic",
                category="tokens",
                units=1,
                op_id=op_id,
                occurred_at=now,
            ),
            daily_cap=1,
        )

    first, second = await asyncio.gather(consume("atomic-1"), consume("atomic-2"))

    assert [first.allowed, second.allowed].count(True) == 1
    assert [first.allowed, second.allowed].count(False) == 1
    assert await ledger.total_for_day("user", "u-atomic", "tokens") == 1
