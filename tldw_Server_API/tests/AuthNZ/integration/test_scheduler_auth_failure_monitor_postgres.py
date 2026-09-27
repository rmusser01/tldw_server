"""Real PostgreSQL coverage for the scheduler's UTC audit-log window."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from loguru import logger

from tldw_Server_API.app.core.AuthNZ import scheduler as scheduler_module
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.settings import get_settings

pytestmark = pytest.mark.integration

NOW = datetime(2026, 9, 16, 0, 3, tzinfo=timezone.utc)
ACTIONS = ("login_failed", "invalid_api_key", "invalid_token")


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz)


@pytest_asyncio.fixture
async def monitor_pool(isolated_test_environment):
    """Use the official isolated database with a pool on this test's loop."""
    pool = DatabasePool(get_settings())
    await pool.initialize()
    try:
        yield pool
    finally:
        await pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("redact", [False, True], ids=["details", "redacted"])
async def test_auth_failure_monitor_postgres_utc_window(monitor_pool, monkeypatch, redact):
    """Count only recent failures; alert above ten and preserve PII policy."""
    pool = monitor_pool
    cutoff = (NOW - timedelta(minutes=5)).replace(tzinfo=None)
    insert = "INSERT INTO audit_logs (action, ip_address, created_at) VALUES ($1, $2, $3)"
    for index in range(10):
        await pool.execute(
            insert,
            ACTIONS[index % 3],
            f"192.0.2.{index % 3 + 1}",
            cutoff + timedelta(microseconds=1),
        )
    for action in ACTIONS:
        await pool.execute(insert, action, "192.0.2.4", cutoff)
        await pool.execute(insert, action, "192.0.2.5", cutoff - timedelta(microseconds=1))
    for action in ("login_success", "metric_auth_failure"):
        await pool.execute(insert, action, "192.0.2.6", NOW.replace(tzinfo=None))

    async def get_pool():
        return pool

    dispatcher = SimpleNamespace(dispatch=AsyncMock(return_value=True))
    monkeypatch.setattr(scheduler_module, "get_db_pool", get_pool)
    monkeypatch.setattr(scheduler_module, "datetime", _FixedDatetime)
    monkeypatch.setattr(scheduler_module, "get_security_alert_dispatcher", lambda: dispatcher)
    scheduler = scheduler_module.AuthNZScheduler()
    scheduler.settings = scheduler.settings.model_copy(update={"PII_REDACT_LOGS": redact})
    warnings = []
    sink_id = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        await scheduler._monitor_auth_failures()
        dispatcher.dispatch.assert_not_awaited()
        assert warnings == []

        await pool.execute(insert, "invalid_token", "192.0.2.1", NOW.replace(tzinfo=None))
        await scheduler._monitor_auth_failures()
    finally:
        logger.remove(sink_id)

    dispatcher.dispatch.assert_awaited_once_with(
        subject="High Authentication Failure Rate",
        message="Details redacted" if redact else "11 failures from 3 IPs",
        severity="high",
        metadata={
            "source": "authnz_scheduler",
            "failure_count": 11,
            "unique_ips": 3,
            "window_minutes": 5,
        },
    )
    assert len(warnings) == 1
    if redact:
        assert "details redacted" in warnings[0]
        assert "11 failures" not in warnings[0]
    else:
        assert "11 failures from 3 unique IPs in last 5 minutes" in warnings[0]
