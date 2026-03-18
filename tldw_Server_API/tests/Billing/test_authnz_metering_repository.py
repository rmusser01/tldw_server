from __future__ import annotations

import sqlite3
from types import TracebackType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


class _AcquireContext:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False


class _FakePool:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self._conn)


class _FakeSqliteConn:
    def __init__(self, execute_side_effect: Any | None = None) -> None:
        self.execute = AsyncMock(side_effect=execute_side_effect)
        self.commit = AsyncMock()


class TestAuthNZUsageDailyRepository:
    @pytest.mark.asyncio
    async def test_fetch_usage_for_date_normalizes_legacy_rows(self):
        from tldw_Server_API.app.core.DB_Management.AuthNZ_Metering_Repository import (
            AuthNZUsageDailyRepository,
        )

        legacy_cursor = MagicMock()
        legacy_cursor.description = [
            ("user_id",),
            ("requests",),
            ("errors",),
            ("bytes_total",),
            ("latency_avg_ms",),
        ]
        legacy_cursor.fetchall = AsyncMock(return_value=[(7, 12, 1, 4096, 25.0)])

        conn = _FakeSqliteConn(
            [
                sqlite3.OperationalError("no such column: bytes_in_total"),
                legacy_cursor,
            ]
        )

        repo = AuthNZUsageDailyRepository(db_pool=_FakePool(conn))
        rows = await repo.fetch_usage_for_date("2026-03-13")

        assert rows == [
            {
                "user_id": 7,
                "requests": 12,
                "errors": 1,
                "bytes_total": 4096,
                "bytes_in_total": 0,
                "latency_avg_ms": 25.0,
            }
        ]


class TestAuthNZBillingSubscriptionRepository:
    @pytest.mark.asyncio
    async def test_get_active_subscription_for_user_falls_back_to_org_owner(self):
        from tldw_Server_API.app.core.DB_Management.AuthNZ_Metering_Repository import (
            AuthNZBillingSubscriptionRepository,
        )

        miss_cursor = MagicMock()
        miss_cursor.description = [
            ("stripe_customer_id",),
            ("stripe_subscription_id",),
            ("org_id",),
        ]
        miss_cursor.fetchall = AsyncMock(return_value=[])

        owner_cursor = MagicMock()
        owner_cursor.description = [
            ("stripe_customer_id",),
            ("stripe_subscription_id",),
            ("org_id",),
        ]
        owner_cursor.fetchall = AsyncMock(return_value=[("cus_owner", "sub_owner", 9)])

        conn = _FakeSqliteConn([miss_cursor, owner_cursor])

        repo = AuthNZBillingSubscriptionRepository(db_pool=_FakePool(conn))
        result = await repo.get_active_subscription_for_user(42)

        assert result == {
            "stripe_customer_id": "cus_owner",
            "stripe_subscription_id": "sub_owner",
            "org_id": 9,
        }

    @pytest.mark.asyncio
    async def test_get_active_subscription_for_user_raises_on_duplicate_memberships(self):
        from tldw_Server_API.app.core.DB_Management.AuthNZ_Metering_Repository import (
            AuthNZBillingSubscriptionRepository,
            DuplicateActiveSubscriptionError,
        )

        duplicate_cursor = MagicMock()
        duplicate_cursor.description = [
            ("stripe_customer_id",),
            ("stripe_subscription_id",),
            ("org_id",),
        ]
        duplicate_cursor.fetchall = AsyncMock(
            return_value=[
                ("cus_a", "sub_a", 1),
                ("cus_b", "sub_b", 2),
            ]
        )

        owner_cursor = MagicMock()
        owner_cursor.description = duplicate_cursor.description
        owner_cursor.fetchall = AsyncMock(return_value=[])

        conn = _FakeSqliteConn([duplicate_cursor, owner_cursor])
        repo = AuthNZBillingSubscriptionRepository(db_pool=_FakePool(conn))

        with pytest.raises(DuplicateActiveSubscriptionError, match="multiple active subscriptions"):
            await repo.get_active_subscription_for_user(42)


class TestAuthNZMeteringSyncLogRepository:
    @pytest.mark.asyncio
    async def test_ensure_schema_creates_sync_log_table_for_sqlite(self):
        from tldw_Server_API.app.core.DB_Management.AuthNZ_Metering_Repository import (
            AuthNZMeteringSyncLogRepository,
        )

        conn = _FakeSqliteConn()
        repo = AuthNZMeteringSyncLogRepository(db_pool=_FakePool(conn))

        await repo.ensure_schema()

        assert conn.execute.await_count == 2
        execute_calls = [call.args[0] for call in conn.execute.await_args_list]
        assert "CREATE TABLE IF NOT EXISTS metering_sync_log" in execute_calls[0]
        assert "CREATE INDEX IF NOT EXISTS idx_metering_sync_log_day" in execute_calls[1]
        conn.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sync_log_repository_checks_records_and_reads_sync_state(self):
        from tldw_Server_API.app.core.DB_Management.AuthNZ_Metering_Repository import (
            AuthNZMeteringSyncLogRepository,
        )

        exists_cursor = MagicMock()
        exists_cursor.fetchone = AsyncMock(return_value=(1,))

        totals_cursor = MagicMock()
        totals_cursor.description = [
            ("user_id",),
            ("stripe_subscription_id",),
            ("requests_synced",),
            ("bytes_synced",),
        ]
        totals_cursor.fetchall = AsyncMock(return_value=[(7, "sub_7", 12, 4096)])

        conn = _FakeSqliteConn([exists_cursor, None, totals_cursor])
        repo = AuthNZMeteringSyncLogRepository(db_pool=_FakePool(conn))

        already_synced = await repo.already_synced(
            user_id=7,
            day="2026-03-13",
            subscription_id="sub_7",
        )
        await repo.record_sync(
            user_id=7,
            day="2026-03-13",
            subscription_id="sub_7",
            requests=12,
            bytes_total=4096,
        )
        rows = await repo.fetch_sync_totals("2026-03-13")

        assert already_synced is True
        assert rows == [
            {
                "user_id": 7,
                "stripe_subscription_id": "sub_7",
                "requests_synced": 12,
                "bytes_synced": 4096,
            }
        ]
        assert conn.commit.await_count == 1
