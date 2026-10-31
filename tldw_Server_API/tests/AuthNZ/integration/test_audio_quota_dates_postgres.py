"""Exercise audio quota DATE parameters through a real PostgreSQL backend."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from types import SimpleNamespace

import asyncpg
import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def audio_postgres(isolated_test_environment, monkeypatch):
    """Use the standard isolated database and a stable UTC quota day."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.Usage import audio_quota

    client, _db_name = isolated_test_environment
    pool = await get_db_pool()
    await audio_quota._ensure_tables(pool)
    monkeypatch.setattr(
        audio_quota,
        "datetime",
        SimpleNamespace(now=lambda _tz: datetime(2026, 9, 10, 12, tzinfo=timezone.utc)),
    )
    monkeypatch.setattr(audio_quota, "_audio_minutes_legacy_backfill_done", False)
    return client, pool


@pytest.mark.asyncio
@pytest.mark.parametrize("minutes_used", [None, 2.5], ids=["no-usage", "existing-usage"])
async def test_postgres_full_profile_reports_current_day_audio_usage(audio_postgres, minutes_used) -> None:
    """The default profile must include real audio quotas without a DATE encoding error."""
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService

    client, pool = audio_postgres
    connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        user_id = await connection.fetchval(
            """
            INSERT INTO users (uuid, username, email, password_hash, is_active, is_verified)
            VALUES ($1, 'audio-profile', 'audio-profile@example.test', $2, TRUE, TRUE)
            RETURNING id
            """,
            uuid.uuid4(),
            PasswordService().hash_password("AudioProfile@Test2026!"),
        )
    finally:
        await connection.close()
    await pool.execute(
        """
        INSERT INTO audio_usage_daily (user_id, day, minutes_used)
        VALUES ($1, DATE '2026-09-09', 9.0)
        """,
        user_id,
    )
    if minutes_used is not None:
        await pool.execute(
            """
            INSERT INTO audio_usage_daily (user_id, day, minutes_used)
            VALUES ($1, DATE '2026-09-10', $2)
            """,
            user_id,
            minutes_used,
        )

    login = client.post(
        "/api/v1/auth/login",
        data={
            "username": "audio-profile",
            "password": "AudioProfile@Test2026!",  # nosec B105
        },
    )
    assert login.status_code == 200
    response = client.get(
        "/api/v1/users/me/profile",
        headers={"Authorization": f"Bearer {login.json()['access_token']}"},
    )

    assert response.status_code == 200
    audio = response.json()["quotas"]["audio"]
    assert audio["daily_minutes_used"] == (0.0 if minutes_used is None else 2.5)
    assert audio["daily_minutes_remaining"] == (30.0 if minutes_used is None else 27.5)


@pytest.mark.asyncio
async def test_postgres_audio_jobs_increment_current_day_without_replacing_minutes(audio_postgres) -> None:
    from tldw_Server_API.app.core.Usage import audio_quota

    _client, pool = audio_postgres
    await pool.execute(
        """
        INSERT INTO audio_usage_daily (user_id, day, minutes_used, jobs_started)
        VALUES (7, DATE '2026-09-09', 9.0, 4)
        """
    )

    await audio_quota.increment_jobs_started(7)
    await pool.execute("UPDATE audio_usage_daily SET minutes_used = 2.5 WHERE user_id = 7 AND day = DATE '2026-09-10'")
    await audio_quota.increment_jobs_started(7)

    rows = await pool.fetch(
        "SELECT day, minutes_used, jobs_started FROM audio_usage_daily WHERE user_id = 7 ORDER BY day"
    )
    assert rows == [
        {"day": date(2026, 9, 9), "minutes_used": 9.0, "jobs_started": 4},
        {"day": date(2026, 9, 10), "minutes_used": 2.5, "jobs_started": 2},
    ]


@pytest.mark.asyncio
async def test_postgres_legacy_audio_backfill_preserves_current_day_usage_once(audio_postgres, monkeypatch) -> None:
    from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import ResourceDailyLedger
    from tldw_Server_API.app.core.Usage import audio_quota

    _client, pool = audio_postgres
    await pool.execute(
        """
        INSERT INTO audio_usage_daily (user_id, day, minutes_used)
        VALUES (7, DATE '2026-09-09', 9.0), (7, DATE '2026-09-10', 2.5), (8, DATE '2026-09-10', 0.0)
        """
    )
    ledger = ResourceDailyLedger(db_pool=pool)
    await ledger.initialize()

    await audio_quota._backfill_audio_usage_daily_to_ledger(ledger)
    # A new process retries the backfill; the existing operation id prevents double counting.
    monkeypatch.setattr(audio_quota, "_audio_minutes_legacy_backfill_done", False)
    await audio_quota._backfill_audio_usage_daily_to_ledger(ledger)

    rows = await pool.fetch("SELECT day_utc, entity_value, units, op_id FROM resource_daily_ledger")
    assert rows == [
        {
            "day_utc": date(2026, 9, 10),
            "entity_value": "7",
            "units": 150,
            "op_id": "audio-minutes-legacy:7:2026-09-10",
        }
    ]
