from datetime import datetime, timezone

import pytest


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_add_daily_minutes_writes_to_resource_daily_ledger(tmp_path, monkeypatch):
    """
    Ensure add_daily_minutes records usage into the generic
    ResourceDailyLedger, which is the canonical source of truth for audio
    daily minutes caps.
    """
    # Point AuthNZ DB to a temporary SQLite file
    db_path = tmp_path / "users_audio_ledger.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.Usage import audio_quota
    from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import ResourceDailyLedger

    # Ensure per-process ledger globals in audio_quota are reset so this test
    # uses the fresh temporary AuthNZ DB configured above.
    audio_quota._daily_ledger = None  # type: ignore[attr-defined]
    audio_quota._audio_minutes_legacy_backfill_done = False  # type: ignore[attr-defined]

    await reset_db_pool()
    pool = await get_db_pool()
    try:
        # Seed minimal audio tables (legacy audio_usage_daily table is present
        # for compatibility but is no longer written to by add_daily_minutes).
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
            CREATE TABLE IF NOT EXISTS audio_user_tiers (
                user_id INTEGER PRIMARY KEY,
                tier TEXT NOT NULL
            )
            """
        )

        user_id = 42
        # Default tier is "free" with a nonzero daily_minutes cap; ledger is
        # the enforcement source of truth.
        await audio_quota.add_daily_minutes(user_id, 2.5)
        await audio_quota.add_daily_minutes(user_id, 2.5)

        # ResourceDailyLedger lives in the same AuthNZ DB; query totals via DAL
        ledger = ResourceDailyLedger(db_pool=pool)
        await ledger.initialize()
        today = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
        # Units are stored as whole seconds; two 2.5 minute events ≈ 300 seconds.
        total_units = await ledger.total_for_day("user", str(user_id), "minutes", day_utc=today)
        assert 295 <= total_units <= 305, f"Expected ~300 seconds, got {total_units}"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_consume_daily_minutes_if_allowed_is_atomic_and_idempotent(tmp_path, monkeypatch):
    db_path = tmp_path / "users_audio_atomic_ledger.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import ResourceDailyLedger
    from tldw_Server_API.app.core.Usage import audio_quota

    audio_quota._daily_ledger = None  # type: ignore[attr-defined]
    audio_quota._audio_minutes_legacy_backfill_done = False  # type: ignore[attr-defined]

    async def _limits(_user_id: int):
        return {
            "daily_minutes": 2.0,
            "concurrent_streams": 1,
            "concurrent_jobs": 1,
            "max_file_size_mb": 25,
        }

    monkeypatch.setattr(audio_quota, "get_limits_for_user", _limits)
    await reset_db_pool()
    pool = await get_db_pool()
    try:
        allowed_1, remaining_1 = await audio_quota.consume_daily_minutes_if_allowed(51, 1.0, op_id="evt-1")
        allowed_2, remaining_2 = await audio_quota.consume_daily_minutes_if_allowed(51, 1.0, op_id="evt-2")
        retry_allowed, retry_remaining = await audio_quota.consume_daily_minutes_if_allowed(51, 1.0, op_id="evt-2")
        denied, remaining_3 = await audio_quota.consume_daily_minutes_if_allowed(51, 0.1, op_id="evt-3")

        assert allowed_1 is True
        assert remaining_1 == pytest.approx(1.0)
        assert allowed_2 is True
        assert remaining_2 == pytest.approx(0.0)
        assert retry_allowed is True
        assert retry_remaining == pytest.approx(0.0)
        assert denied is False
        assert remaining_3 == pytest.approx(0.0)

        ledger = ResourceDailyLedger(db_pool=pool)
        await ledger.initialize()
        today = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
        assert await ledger.total_for_day("user", "51", "minutes", day_utc=today) == 120
    finally:
        await pool.close()

