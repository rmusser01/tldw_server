"""
Tests for Stripe usage metering reconciliation service.

Covers:
- Service initialization and configuration
- Billing-disabled skip behavior
- sync_daily_usage with mocked DB and Stripe calls
- check_reconciliation with mocked DB
- Default date handling (yesterday)
- Double-counting prevention
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldw_Server_API.app.services import stripe_metering_service as _sms_module
from tldw_Server_API.app.services.stripe_metering_service import StripeMeteringService


def _stripe_ok():
    """Pretend Stripe is installed and provide a mock module when needed."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch.object(_sms_module, "STRIPE_AVAILABLE", True))
    if _sms_module.stripe is None:
        fake_stripe = MagicMock()
        fake_stripe.api_key = None
        stack.enter_context(patch.object(_sms_module, "stripe", fake_stripe))
    return stack


@pytest.fixture
def mock_stripe() -> MagicMock:
    mock = MagicMock()
    mock.api_key = None
    return mock


@pytest.fixture
def stripe_enabled(monkeypatch: pytest.MonkeyPatch, mock_stripe: MagicMock) -> MagicMock:
    monkeypatch.setattr(_sms_module, "STRIPE_AVAILABLE", True)
    monkeypatch.setattr(_sms_module, "stripe", mock_stripe)
    return mock_stripe


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestStripeMeteringServiceInit:
    """Tests for StripeMeteringService initialization."""

    def test_disabled_by_default(self):
        """Service is disabled when no env vars are set."""
        with patch.dict("os.environ", {}, clear=True):
            svc = StripeMeteringService()
            assert svc.is_enabled is False

    def test_enabled_with_env_vars(self):
        """Service is enabled when both BILLING_ENABLED and STRIPE_API_KEY are set."""
        with patch.dict(
            "os.environ",
            {"BILLING_ENABLED": "true", "STRIPE_API_KEY": "sk_test_123"},
        ), patch.object(_sms_module, "STRIPE_AVAILABLE", True):
            svc = StripeMeteringService()
            assert svc.is_enabled is True

    def test_enabled_with_numeric_flag(self):
        """BILLING_ENABLED=1 also enables the service."""
        with patch.dict(
            "os.environ",
            {"BILLING_ENABLED": "1", "STRIPE_API_KEY": "sk_test_abc"},
        ), patch.object(_sms_module, "STRIPE_AVAILABLE", True):
            svc = StripeMeteringService()
            assert svc.is_enabled is True

    def test_not_enabled_without_stripe_key(self):
        """Service is not enabled if STRIPE_API_KEY is missing."""
        with patch.dict("os.environ", {"BILLING_ENABLED": "true"}, clear=True):
            svc = StripeMeteringService()
            assert svc.is_enabled is False

    def test_not_enabled_when_stripe_package_is_unavailable(self):
        with patch.dict(
            "os.environ",
            {"BILLING_ENABLED": "true", "STRIPE_API_KEY": "sk_test_123"},
            clear=True,
        ), patch.object(_sms_module, "STRIPE_AVAILABLE", False):
            svc = StripeMeteringService()
            assert svc.is_enabled is False


# ---------------------------------------------------------------------------
# sync_daily_usage tests
# ---------------------------------------------------------------------------

class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        base = datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc)
        if tz is None:
            return base.replace(tzinfo=None)
        return base.astimezone(tz)


@pytest.fixture
def enabled_service() -> StripeMeteringService:
    with patch.dict(
        "os.environ",
        {"BILLING_ENABLED": "true", "STRIPE_API_KEY": "sk_test_123"},
    ):
        return StripeMeteringService()


@pytest.fixture
def frozen_utc_now(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(_sms_module, "datetime", _FrozenDateTime)
    return "2026-03-13"


@pytest.fixture
def fake_pool():
    def _make(conn):
        return _FakePool(conn)

    return _make


@pytest.fixture
def fake_sqlite_conn():
    def _make(execute_side_effect):
        return _FakeSqliteConn(execute_side_effect)

    return _make


@pytest.fixture
def fake_postgres_conn() -> _FakePostgresConn:
    return _FakePostgresConn()


class _AcquireContext:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireContext(self._conn)


class _FakePostgresConn:
    def __init__(self):
        self.fetch_calls = []
        self.fetchval_calls = []
        self.execute_calls = []
        self.fetchrow_calls = []

    async def fetch(self, query, *params):
        self.fetch_calls.append((query, params))
        return []

    async def fetchrow(self, query, *params):
        self.fetchrow_calls.append((query, params))
        return None

    async def fetchval(self, query, *params):
        self.fetchval_calls.append((query, params))
        return None

    async def execute(self, query, *params):
        self.execute_calls.append((query, params))
        return None


class _FakeSqliteConn:
    def __init__(self, execute_side_effect):
        self.execute = AsyncMock(side_effect=execute_side_effect)


class TestSyncDailyUsage:
    """Tests for StripeMeteringService.sync_daily_usage."""

    def test_rejects_partial_repository_injection_without_db_pool(self):
        from tldw_Server_API.app.services.stripe_metering_service import StripeMeteringService

        with pytest.raises(
            ValueError,
            match="Inject all metering repositories together or provide db_pool",
        ):
            StripeMeteringService(usage_repo=MagicMock())

    @pytest.mark.asyncio
    async def test_uses_injected_repositories_for_sync(self):
        with patch.dict(
            "os.environ",
            {"BILLING_ENABLED": "true", "STRIPE_API_KEY": "sk_test_123"},
        ):
            svc = StripeMeteringService(
                usage_repo=MagicMock(),
                subscription_repo=MagicMock(),
                sync_log_repo=MagicMock(),
            )
        svc._usage_repo.fetch_usage_for_date = AsyncMock(return_value=[
            {
                "user_id": 1,
                "requests": 100,
                "errors": 0,
                "bytes_total": 5000,
                "bytes_in_total": 3000,
                "latency_avg_ms": 50.0,
            },
        ])
        svc._subscription_repo.get_active_subscription_for_user = AsyncMock(return_value={
            "stripe_customer_id": "cus_test1",
            "stripe_subscription_id": "sub_test1",
            "org_id": 10,
        })
        svc._sync_log_repo.ensure_schema = AsyncMock()
        svc._sync_log_repo.already_synced = AsyncMock(return_value=False)
        svc._sync_log_repo.record_sync = AsyncMock()
        svc._get_subscription_metered_item = AsyncMock(return_value="si_item1")
        svc._report_usage_to_stripe = AsyncMock()

        with _stripe_ok():
            result = await svc.sync_daily_usage(date="2026-03-13")

        assert result["status"] == "completed"
        assert result["synced_users"] == 1
        svc._sync_log_repo.ensure_schema.assert_awaited_once()
        svc._usage_repo.fetch_usage_for_date.assert_awaited_once_with("2026-03-13")
        svc._subscription_repo.get_active_subscription_for_user.assert_awaited_once_with(1)
        svc._sync_log_repo.already_synced.assert_awaited_once_with(
            user_id=1,
            day="2026-03-13",
            subscription_id="sub_test1",
        )
        svc._sync_log_repo.record_sync.assert_awaited_once_with(
            user_id=1,
            day="2026-03-13",
            subscription_id="sub_test1",
            requests=100,
            bytes_total=5000,
        )

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self):
        """Returns skip status when billing is not enabled."""
        with patch.dict("os.environ", {}, clear=True):
            svc = StripeMeteringService()
            result = await svc.sync_daily_usage()

        assert result["status"] == "skipped"
        assert result["reason"] == "billing_not_enabled"

    @pytest.mark.asyncio
    async def test_skips_when_stripe_not_installed(self, enabled_service):
        """Returns skip status when stripe package is not installed."""
        svc = enabled_service
        with patch.object(_sms_module, "STRIPE_AVAILABLE", False):
            result = await svc.sync_daily_usage(date="2026-03-13")

        assert result["status"] == "skipped"
        assert result["reason"] == "stripe_package_not_installed"

    @pytest.mark.asyncio
    async def test_returns_completed_no_usage(self, enabled_service, stripe_enabled):
        """Returns completed with zero synced_users when no usage data exists."""
        svc = enabled_service
        svc._get_db_pool = AsyncMock(return_value=MagicMock())
        svc._ensure_metering_sync_table = AsyncMock()
        svc._query_usage_for_date = AsyncMock(return_value=[])

        result = await svc.sync_daily_usage(date="2026-03-13")

        assert result["status"] == "completed"
        assert result["date"] == "2026-03-13"
        assert result["synced_users"] == 0
        assert result["errors"] == 0
        assert result["message"] == "no_usage_data"

    @pytest.mark.asyncio
    async def test_defaults_to_yesterday(self, enabled_service, frozen_utc_now, stripe_enabled):
        """When no date is specified, defaults to yesterday."""
        svc = enabled_service
        svc._get_db_pool = AsyncMock(return_value=MagicMock())
        svc._ensure_metering_sync_table = AsyncMock()
        svc._query_usage_for_date = AsyncMock(return_value=[])

        result = await svc.sync_daily_usage()

        assert result["date"] == frozen_utc_now

    @pytest.mark.asyncio
    async def test_syncs_user_with_subscription(self, enabled_service, stripe_enabled):
        """Successfully syncs usage for a user with an active Stripe subscription."""
        svc = enabled_service
        svc._get_db_pool = AsyncMock(return_value=MagicMock())
        svc._ensure_metering_sync_table = AsyncMock()
        svc._query_usage_for_date = AsyncMock(return_value=[
            {"user_id": 1, "requests": 100, "errors": 2, "bytes_total": 5000,
             "bytes_in_total": 3000, "latency_avg_ms": 50.0},
        ])
        svc._query_user_subscription = AsyncMock(return_value={
            "stripe_customer_id": "cus_test1",
            "stripe_subscription_id": "sub_test1",
            "org_id": 10,
        })
        svc._already_synced = AsyncMock(return_value=False)
        svc._get_subscription_metered_item = AsyncMock(return_value="si_item1")
        svc._report_usage_to_stripe = AsyncMock()
        svc._record_sync = AsyncMock()

        result = await svc.sync_daily_usage(date="2026-03-13")

        assert result["status"] == "completed"
        assert result["synced_users"] == 1
        assert result["errors"] == 0

        svc._report_usage_to_stripe.assert_called_once()
        call_args = svc._report_usage_to_stripe.call_args
        assert call_args[0][0] == "si_item1"
        assert call_args[0][1] == 100  # requests quantity
        svc._record_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_already_synced_user(self, enabled_service, stripe_enabled):
        """Skips users whose usage was already synced (prevents double-counting)."""
        svc = enabled_service
        svc._get_db_pool = AsyncMock(return_value=MagicMock())
        svc._ensure_metering_sync_table = AsyncMock()
        svc._query_usage_for_date = AsyncMock(return_value=[
            {"user_id": 1, "requests": 50, "errors": 0, "bytes_total": 1000,
             "bytes_in_total": 500, "latency_avg_ms": 30.0},
        ])
        svc._query_user_subscription = AsyncMock(return_value={
            "stripe_customer_id": "cus_test1",
            "stripe_subscription_id": "sub_test1",
            "org_id": 10,
        })
        svc._already_synced = AsyncMock(return_value=True)

        result = await svc.sync_daily_usage(date="2026-03-13")

        assert result["synced_users"] == 0
        assert result["skipped_users"] == 1

    @pytest.mark.asyncio
    async def test_skips_user_without_subscription(self, enabled_service, stripe_enabled):
        """Skips users who have no active Stripe subscription."""
        svc = enabled_service
        svc._get_db_pool = AsyncMock(return_value=MagicMock())
        svc._ensure_metering_sync_table = AsyncMock()
        svc._query_usage_for_date = AsyncMock(return_value=[
            {"user_id": 99, "requests": 10, "errors": 0, "bytes_total": 100,
             "bytes_in_total": 50, "latency_avg_ms": 10.0},
        ])
        svc._query_user_subscription = AsyncMock(return_value=None)

        result = await svc.sync_daily_usage(date="2026-03-13")

        assert result["synced_users"] == 0
        assert result["skipped_users"] == 1

    @pytest.mark.asyncio
    async def test_skips_user_with_zero_requests(self, enabled_service, stripe_enabled):
        """Skips users with zero requests."""
        svc = enabled_service
        svc._get_db_pool = AsyncMock(return_value=MagicMock())
        svc._ensure_metering_sync_table = AsyncMock()
        svc._query_usage_for_date = AsyncMock(return_value=[
            {"user_id": 1, "requests": 0, "errors": 0, "bytes_total": 0,
             "bytes_in_total": 0, "latency_avg_ms": 0},
        ])

        result = await svc.sync_daily_usage(date="2026-03-13")

        assert result["synced_users"] == 0
        assert result["skipped_users"] == 1

    @pytest.mark.asyncio
    async def test_handles_stripe_error(self, enabled_service, stripe_enabled):
        """Counts errors when Stripe API call fails."""
        svc = enabled_service
        svc._get_db_pool = AsyncMock(return_value=MagicMock())
        svc._ensure_metering_sync_table = AsyncMock()
        svc._query_usage_for_date = AsyncMock(return_value=[
            {"user_id": 1, "requests": 100, "errors": 0, "bytes_total": 5000,
             "bytes_in_total": 3000, "latency_avg_ms": 50.0},
        ])
        svc._query_user_subscription = AsyncMock(return_value={
            "stripe_customer_id": "cus_test1",
            "stripe_subscription_id": "sub_test1",
            "org_id": 10,
        })
        svc._already_synced = AsyncMock(return_value=False)
        svc._get_subscription_metered_item = AsyncMock(return_value="si_item1")
        svc._report_usage_to_stripe = AsyncMock(
            side_effect=RuntimeError("Stripe API error")
        )

        result = await svc.sync_daily_usage(date="2026-03-13")

        assert result["errors"] == 1
        assert result["synced_users"] == 0

    @pytest.mark.asyncio
    async def test_db_pool_error_returns_error_status(self, enabled_service, stripe_enabled):
        """Returns error status when DB pool is unavailable."""
        svc = enabled_service
        svc._get_db_pool = AsyncMock(side_effect=RuntimeError("no pool"))

        result = await svc.sync_daily_usage(date="2026-03-13")

        assert result["status"] == "error"
        assert "db_pool_unavailable" in result["error"]

    @pytest.mark.asyncio
    async def test_skips_user_without_metered_item(self, enabled_service, stripe_enabled):
        """Skips users whose subscription has no metered item."""
        svc = enabled_service
        svc._get_db_pool = AsyncMock(return_value=MagicMock())
        svc._ensure_metering_sync_table = AsyncMock()
        svc._query_usage_for_date = AsyncMock(return_value=[
            {"user_id": 1, "requests": 100, "errors": 0, "bytes_total": 5000,
             "bytes_in_total": 3000, "latency_avg_ms": 50.0},
        ])
        svc._query_user_subscription = AsyncMock(return_value={
            "stripe_customer_id": "cus_test1",
            "stripe_subscription_id": "sub_test1",
            "org_id": 10,
        })
        svc._already_synced = AsyncMock(return_value=False)
        svc._get_subscription_metered_item = AsyncMock(return_value=None)

        result = await svc.sync_daily_usage(date="2026-03-13")

        assert result["synced_users"] == 0
        assert result["skipped_users"] == 1

    @pytest.mark.asyncio
    async def test_query_usage_for_date_falls_back_when_bytes_in_total_missing(self, enabled_service, fake_pool, fake_sqlite_conn):
        svc = enabled_service
        legacy_cursor = MagicMock()
        legacy_cursor.description = [
            ("user_id",),
            ("requests",),
            ("errors",),
            ("bytes_total",),
            ("latency_avg_ms",),
        ]
        legacy_cursor.fetchall = AsyncMock(return_value=[(7, 12, 1, 4096, 25.0)])

        conn = fake_sqlite_conn(
            [
                sqlite3.OperationalError("no such column: bytes_in_total"),
                legacy_cursor,
            ]
        )

        rows = await svc._query_usage_for_date(fake_pool(conn), "2026-03-13")

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

    @pytest.mark.asyncio
    async def test_query_user_subscription_falls_back_to_org_owner(self, enabled_service, fake_pool, fake_sqlite_conn):
        svc = enabled_service
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

        conn = fake_sqlite_conn([miss_cursor, owner_cursor])

        result = await svc._query_user_subscription(fake_pool(conn), 42)

        assert result == {
            "stripe_customer_id": "cus_owner",
            "stripe_subscription_id": "sub_owner",
            "org_id": 9,
        }

    @pytest.mark.asyncio
    async def test_postgres_usage_queries_bind_real_date_objects(self, enabled_service, fake_pool, fake_postgres_conn):
        svc = enabled_service
        conn = fake_postgres_conn
        pool = fake_pool(conn)

        await svc._query_usage_for_date(pool, "2026-03-13")
        await svc._query_sync_totals(pool, "2026-03-13")

        assert isinstance(conn.fetch_calls[0][1][0], date)
        assert isinstance(conn.fetch_calls[1][1][0], date)

    @pytest.mark.asyncio
    async def test_postgres_sync_log_helpers_bind_real_date_objects(self, enabled_service, fake_pool, fake_postgres_conn):
        svc = enabled_service
        conn = fake_postgres_conn
        pool = fake_pool(conn)

        await svc._already_synced(pool, 7, "2026-03-13", "sub_123")
        await svc._record_sync(pool, 7, "2026-03-13", "sub_123", 11, 2048)

        assert isinstance(conn.fetchval_calls[0][1][1], date)
        assert isinstance(conn.execute_calls[0][1][1], date)

    @pytest.mark.asyncio
    async def test_get_subscription_metered_item_returns_none_for_missing_subscription(self, enabled_service):
        svc = enabled_service

        class _MissingStripeResource(Exception):
            code = "resource_missing"

        fake_stripe = MagicMock()
        fake_stripe.Subscription.retrieve.side_effect = _MissingStripeResource(
            "No such subscription: 'sub_missing'"
        )

        with patch.object(_sms_module, "STRIPE_AVAILABLE", True), patch.object(
            _sms_module, "stripe", fake_stripe
        ):
            assert await svc._get_subscription_metered_item("sub_missing") is None

    @pytest.mark.asyncio
    async def test_get_subscription_metered_item_propagates_non_missing_stripe_errors(self, enabled_service):
        svc = enabled_service

        class _StripeAPIError(Exception):
            code = "api_error"

        fake_stripe = MagicMock()
        fake_stripe.Subscription.retrieve.side_effect = _StripeAPIError("rate limited")

        with patch.object(_sms_module, "STRIPE_AVAILABLE", True), patch.object(
            _sms_module, "stripe", fake_stripe
        ):
            with pytest.raises(_StripeAPIError, match="rate limited"):
                await svc._get_subscription_metered_item("sub_problem")


# ---------------------------------------------------------------------------
# check_reconciliation tests
# ---------------------------------------------------------------------------


class TestCheckReconciliation:
    """Tests for StripeMeteringService.check_reconciliation."""

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self):
        """Returns skip status when billing is not enabled."""
        with patch.dict("os.environ", {}, clear=True):
            svc = StripeMeteringService()
            result = await svc.check_reconciliation()

        assert result["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_returns_completed_no_discrepancies(self, enabled_service):
        """Returns completed with no discrepancies when usage matches synced data."""
        svc = enabled_service
        svc._get_db_pool = AsyncMock(return_value=MagicMock())
        svc._query_usage_for_date = AsyncMock(return_value=[
            {"user_id": 1, "requests": 100, "errors": 0, "bytes_total": 5000,
             "bytes_in_total": 3000, "latency_avg_ms": 50.0},
        ])
        svc._query_sync_totals = AsyncMock(return_value=[
            {"user_id": 1, "stripe_subscription_id": "sub_1",
             "requests_synced": 100, "bytes_synced": 5000},
        ])

        result = await svc.check_reconciliation(date="2026-03-13")

        assert result["status"] == "completed"
        assert result["date"] == "2026-03-13"
        assert isinstance(result["discrepancies"], list)
        assert len(result["discrepancies"]) == 0
        assert result["total_local_requests"] == 100
        assert result["total_synced_requests"] == 100

    @pytest.mark.asyncio
    async def test_detects_drift(self, enabled_service):
        """Detects discrepancies between local usage and synced records."""
        svc = enabled_service
        svc._get_db_pool = AsyncMock(return_value=MagicMock())
        svc._query_usage_for_date = AsyncMock(return_value=[
            {"user_id": 1, "requests": 100, "errors": 0, "bytes_total": 5000,
             "bytes_in_total": 3000, "latency_avg_ms": 50.0},
            {"user_id": 2, "requests": 50, "errors": 0, "bytes_total": 1000,
             "bytes_in_total": 500, "latency_avg_ms": 20.0},
        ])
        svc._query_sync_totals = AsyncMock(return_value=[
            {"user_id": 1, "stripe_subscription_id": "sub_1",
             "requests_synced": 80, "bytes_synced": 4000},
            # user_id 2 has no sync record
        ])

        result = await svc.check_reconciliation(date="2026-03-13")

        assert result["status"] == "completed"
        assert len(result["discrepancies"]) == 2
        # User 1: 100 local, 80 synced => drift of 20
        d1 = next(d for d in result["discrepancies"] if d["user_id"] == 1)
        assert d1["drift"] == 20
        # User 2: 50 local, 0 synced => drift of 50
        d2 = next(d for d in result["discrepancies"] if d["user_id"] == 2)
        assert d2["drift"] == 50

    @pytest.mark.asyncio
    async def test_handles_sync_log_table_missing(self, enabled_service):
        """Gracefully handles missing metering_sync_log table."""
        svc = enabled_service
        svc._get_db_pool = AsyncMock(return_value=MagicMock())
        svc._query_usage_for_date = AsyncMock(return_value=[
            {"user_id": 1, "requests": 50, "errors": 0, "bytes_total": 1000,
             "bytes_in_total": 500, "latency_avg_ms": 10.0},
        ])
        svc._query_sync_totals = AsyncMock(
            side_effect=Exception("no such table: metering_sync_log")
        )

        result = await svc.check_reconciliation(date="2026-03-13")

        assert result["status"] == "completed"
        # All usage shows as drift since no sync records
        assert len(result["discrepancies"]) == 1
        assert result["discrepancies"][0]["drift"] == 50

    @pytest.mark.asyncio
    async def test_defaults_to_yesterday(self, enabled_service, frozen_utc_now):
        """When no date is specified, defaults to yesterday."""
        svc = enabled_service
        svc._get_db_pool = AsyncMock(return_value=MagicMock())
        svc._query_usage_for_date = AsyncMock(return_value=[])
        svc._query_sync_totals = AsyncMock(return_value=[])

        result = await svc.check_reconciliation()

        assert result["date"] == frozen_utc_now

    @pytest.mark.asyncio
    async def test_db_pool_error_returns_error(self, enabled_service):
        """Returns error status when DB pool is unavailable."""
        svc = enabled_service
        svc._get_db_pool = AsyncMock(side_effect=RuntimeError("no pool"))

        result = await svc.check_reconciliation(date="2026-03-13")

        assert result["status"] == "error"
        assert "db_pool_unavailable" in result["error"]
        assert isinstance(result["discrepancies"], list)
