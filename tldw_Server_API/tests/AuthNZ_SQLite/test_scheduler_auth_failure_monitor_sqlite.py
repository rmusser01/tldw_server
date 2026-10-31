"""SQLite controls for the scheduler's existing ISO timestamp boundary."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ import scheduler as scheduler_module
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.settings import Settings

pytestmark = pytest.mark.integration

NOW = datetime(2026, 9, 16, 0, 3, tzinfo=timezone.utc)
ACTIONS = ("login_failed", "invalid_api_key", "invalid_token")


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz)


@pytest.mark.asyncio
@pytest.mark.parametrize("redact", [False, True], ids=["details", "redacted"])
async def test_auth_failure_monitor_sqlite_utc_window(tmp_path, monkeypatch, redact):
    """Keep the SQLite cutoff, action filters, threshold and PII policy."""
    settings = Settings(
        AUTH_MODE="single_user",
        DATABASE_URL=f"sqlite:///{tmp_path / 'monitor.sqlite'}",
        PII_REDACT_LOGS=redact,
    )
    pool = DatabasePool(settings)
    await pool.initialize()
    try:
        cutoff = NOW - timedelta(minutes=5)
        insert = "INSERT INTO audit_logs (action, ip_address, created_at) VALUES (?, ?, ?)"
        for index in range(10):
            await pool.execute(
                insert,
                (ACTIONS[index % 3], f"192.0.2.{index % 3 + 1}", (cutoff + timedelta(microseconds=1)).isoformat()),
            )
        for action in ACTIONS:
            await pool.execute(insert, (action, "192.0.2.4", cutoff.isoformat()))
            await pool.execute(insert, (action, "192.0.2.5", (cutoff - timedelta(microseconds=1)).isoformat()))
        for action in ("login_success", "metric_auth_failure"):
            await pool.execute(insert, (action, "192.0.2.6", NOW.isoformat()))

        async def get_pool():
            return pool

        dispatcher = SimpleNamespace(dispatch=AsyncMock(return_value=True))
        monkeypatch.setattr(scheduler_module, "get_db_pool", get_pool)
        monkeypatch.setattr(scheduler_module, "datetime", _FixedDatetime)
        monkeypatch.setattr(scheduler_module, "get_security_alert_dispatcher", lambda: dispatcher)
        scheduler = scheduler_module.AuthNZScheduler()
        scheduler.settings = settings
        warnings = []
        sink_id = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
        try:
            await scheduler._monitor_auth_failures()
            dispatcher.dispatch.assert_not_awaited()
            assert warnings == []

            await pool.execute(insert, ("invalid_token", "192.0.2.1", NOW.isoformat()))
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
    finally:
        await pool.close()
