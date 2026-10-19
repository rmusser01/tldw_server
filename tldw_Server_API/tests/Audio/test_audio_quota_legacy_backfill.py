from datetime import datetime, timezone

import pytest
from tldw_Server_API.tests.helpers.audio_quota_state import cold_audio_ledger as cold_audio_ledger


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_legacy_audio_usage_daily_backfilled_into_resource_daily_ledger(tmp_path, monkeypatch):
    """
    Simulate an upgrade where audio_usage_daily already contains usage for
    today and verify that the first ResourceDailyLedger initialization
    backfills that usage so daily caps remain accurate.
    """
    db_path = tmp_path / "users_audio_legacy.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.Usage import audio_quota
    from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import ResourceDailyLedger

    await reset_db_pool()
    pool = await get_db_pool()
    try:
        # Seed legacy audio_usage_daily with prior usage for today
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_usage_daily (
                user_id INTEGER NOT NULL,
                day TEXT NOT NULL,
                minutes_used REAL NOT NULL DEFAULT 0,
                jobs_started INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, day)
            )
            """
        )
        await pool.execute(
            """
            INSERT INTO audio_usage_daily (user_id, day, minutes_used, jobs_started)
            VALUES (?, ?, ?, 0)
            """,
            7,
            datetime.now(timezone.utc).date().strftime("%Y-%m-%d"),
            2.5,
        )

        # Reset any cached ledger state in the module
        audio_quota._daily_ledger = None  # type: ignore[attr-defined]
        audio_quota._audio_minutes_legacy_backfill_done = False  # type: ignore[attr-defined]

        # Trigger ledger initialization + backfill
        ledger = await audio_quota._get_daily_ledger()
        assert ledger is not None

        # Verify ResourceDailyLedger now reflects the legacy usage
        check_ledger = ResourceDailyLedger(db_pool=pool)
        await check_ledger.initialize()
        today = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
        total_units = await check_ledger.total_for_day("user", "7", "minutes", day_utc=today)
        # 2.5 minutes ≈ 150 seconds
        assert 145 <= total_units <= 155, f"Expected ~150 seconds from backfill, got {total_units}"
    finally:
        await pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_period", ["daily", "monthly"])
async def test_reporting_first_backfills_legacy_audio_minutes(tmp_path, monkeypatch, first_period, cold_audio_ledger):
    """A reporting-first upgrade includes legacy usage before any admission."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.Usage import audio_quota, quota_checks

    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'reporting_first.db'}")
    await reset_db_pool()
    pool = await get_db_pool()
    cold_audio_ledger()
    try:
        await audio_quota._ensure_tables(pool)
        await pool.execute(
            "INSERT INTO audio_usage_daily (user_id, day, minutes_used, jobs_started) VALUES (?, ?, ?, 0)",
            7,
            datetime.now(timezone.utc).date().isoformat(),
            2.5,
        )
        read = audio_quota.get_daily_minutes_used if first_period == "daily" else audio_quota.get_monthly_minutes_used
        assert await read(7) == 2.5  # nosec B101 - pytest regression assertion.
        assert await audio_quota.get_daily_minutes_used(7) == 2.5  # nosec B101 - pytest regression assertion.
        assert await audio_quota.get_monthly_minutes_used(7) == 2.5  # nosec B101 - pytest regression assertion.
        # A fresh process repeats the existing idempotent migration without doubling usage.
        cold_audio_ledger()
        assert await read(7) == 2.5  # nosec B101 - pytest regression assertion.
    finally:
        quota_checks.reset_ledger_cache()
        await reset_db_pool()
