"""Exercise audio quota DATE parameters through a real PostgreSQL backend."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import asyncpg
import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def audio_postgres(isolated_test_environment, monkeypatch):
    """Use the standard isolated database and a stable UTC quota day."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.Usage import audio_quota, quota_checks, quota_resolver

    client, _db_name = isolated_test_environment
    pool = await get_db_pool()
    await audio_quota._ensure_tables(pool)
    monkeypatch.setattr(
        audio_quota,
        "datetime",
        SimpleNamespace(now=lambda _tz: datetime(2026, 9, 10, 12, tzinfo=timezone.utc)),
    )
    monkeypatch.setattr(audio_quota, "_audio_minutes_legacy_backfill_done", False)
    monkeypatch.setattr(audio_quota, "_daily_ledger", None)
    quota_checks.reset_ledger_cache()
    # Per-test databases reuse user IDs, but the resolver cache is process-local.
    quota_resolver.invalidate_all()
    try:
        yield client, pool
    finally:
        quota_resolver.invalidate_all()
        quota_checks.reset_ledger_cache()


@pytest_asyncio.fixture
async def audio_profile_user(audio_postgres, minutes_used, monkeypatch):
    """Create the actual profile user and retain previous/current-day usage."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
    from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import LedgerEntry, ResourceDailyLedger
    from tldw_Server_API.app.core.Usage import quota_checks, quota_resolver

    client, fixture_pool = audio_postgres
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")

    async def seed_profile_user():
        """Keep pooled profile setup on the same loop as the actual portal."""
        pool = await get_db_pool()
        assert pool.settings.DATABASE_URL == fixture_pool.settings.DATABASE_URL
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
        # Reported minutes come from the resource ledger's real UTC day.
        quota_checks.reset_ledger_cache()
        ledger = ResourceDailyLedger(db_pool=pool)
        await ledger.initialize()
        now = datetime.now(timezone.utc)
        seed = [(now - timedelta(days=1), 9.0, "yesterday")]
        if minutes_used is not None:
            seed.append((now, minutes_used, "today"))
        for occurred_at, minutes, tag in seed:
            await ledger.add(
                LedgerEntry(
                    entity_scope="user",
                    entity_value=str(user_id),
                    category="minutes",
                    units=int(minutes * 60),
                    op_id=f"audio-profile-test:{user_id}:{tag}",
                    occurred_at=occurred_at,
                )
            )
        quota_resolver.invalidate_user(user_id)
        return pool, user_id

    pool, user_id = client.portal.call(seed_profile_user)
    try:
        login = client.post(
            "/api/v1/auth/login",
            data={
                "username": "audio-profile",
                "password": "AudioProfile@Test2026!",  # nosec B105
            },
        )
        assert login.status_code == 200
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        assert client.portal.call(get_db_pool) is pool
        yield client, pool, user_id, headers
    finally:
        quota_resolver.invalidate_user(user_id)
        quota_checks.reset_ledger_cache()


@pytest_asyncio.fixture
async def configured_audio_profile_user(audio_profile_user, daily_limit):
    """Optionally persist a limit and invalidate this user's primed unlimited cache."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.Usage import quota_resolver
    from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo

    client, pool, user_id, _headers = audio_profile_user

    async def configure_limit():
        """Prime and reread limits on the profile request's owning loop."""
        assert await get_db_pool() is pool
        repo = UserProfileOverridesRepo(pool)
        await repo.ensure_tables()
        assert await quota_resolver.user_quota(user_id, "limits.audio_daily_minutes") is None
        if daily_limit is not None:
            await repo.upsert_override(
                user_id=user_id, key="limits.audio_daily_minutes", value=daily_limit, updated_by=user_id,
            )
            overrides = await repo.list_overrides_for_user(user_id)
            assert [(row["key"], row["value"]) for row in overrides] == [("limits.audio_daily_minutes", daily_limit)]
            assert await quota_resolver.user_quota(user_id, "limits.audio_daily_minutes") is None
        else:
            assert await repo.list_overrides_for_user(user_id) == []
        quota_resolver.invalidate_user(user_id)
        assert await quota_resolver.user_quota(user_id, "limits.audio_daily_minutes") == daily_limit

    async def cleanup_limit():
        """Delete only an override this fixture was configured to create."""
        try:
            if daily_limit is not None:
                cleanup_pool = await get_db_pool()
                assert cleanup_pool.settings.DATABASE_URL == pool.settings.DATABASE_URL
                await UserProfileOverridesRepo(cleanup_pool).delete_override(
                    user_id=user_id, key="limits.audio_daily_minutes"
                )
        finally:
            quota_resolver.invalidate_user(user_id)

    try:
        client.portal.call(configure_limit)
        yield audio_profile_user
    finally:
        client.portal.call(cleanup_limit)


@pytest.mark.asyncio
@pytest.mark.parametrize("minutes_used", [None, 2.5], ids=["no-usage", "existing-usage"])
@pytest.mark.parametrize(
    ("quotas_enabled", "daily_limit"),
    [(False, None), (True, None), (True, 30.0)],
    ids=["quotas-off", "no-override", "configured-limit"],
)
async def test_postgres_full_profile_reports_current_day_audio_usage(
    configured_audio_profile_user, minutes_used, quotas_enabled, daily_limit, monkeypatch,
) -> None:
    """Report DATE-backed usage with unlimited or explicitly configured quotas."""
    client, _pool, _user_id, headers = configured_audio_profile_user
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1" if quotas_enabled else "0")
    response = client.get(
        "/api/v1/users/me/profile",
        headers=headers,
    )

    assert response.status_code == 200
    audio = response.json()["quotas"]["audio"]
    assert audio["daily_minutes_used"] == (0.0 if minutes_used is None else 2.5)
    expected_remaining = None if daily_limit is None else daily_limit - (minutes_used or 0.0)
    assert audio["daily_minutes_remaining"] == expected_remaining


@pytest.mark.asyncio
@pytest.mark.parametrize("minutes_used", [None, 2.5], ids=["no-usage", "existing-usage"])
@pytest.mark.parametrize("daily_limit", [30.0], ids=["configured-limit"])
async def test_postgres_audio_profile_disabled_quota_preserves_usage(
    configured_audio_profile_user, minutes_used, monkeypatch,
) -> None:
    """Disabling quotas reports usage but no remaining cap despite a persisted limit."""
    client, _pool, _user_id, headers = configured_audio_profile_user
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "0")

    response = client.get("/api/v1/users/me/profile", headers=headers)

    assert response.status_code == 200
    audio = response.json()["quotas"]["audio"]
    assert audio["daily_minutes_used"] == (0.0 if minutes_used is None else 2.5)
    assert audio["daily_minutes_remaining"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("minutes_used", [None, 2.5], ids=["no-usage", "existing-usage"])
async def test_postgres_audio_profile_without_limit_preserves_usage(audio_profile_user, minutes_used) -> None:
    """Enabled quotas without a configured limit remain unlimited, with real DATE usage."""
    from tldw_Server_API.app.core.Usage import quota_resolver
    from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo

    client, pool, user_id, headers = audio_profile_user

    async def verify_unlimited():
        """Read the real profile overrides without replacing the portal's pool."""
        assert await UserProfileOverridesRepo(pool).list_overrides_for_user(user_id) == []
        quota_resolver.invalidate_user(user_id)

    client.portal.call(verify_unlimited)

    response = client.get("/api/v1/users/me/profile", headers=headers)

    assert response.status_code == 200
    audio = response.json()["quotas"]["audio"]
    assert audio["daily_minutes_used"] == (0.0 if minutes_used is None else 2.5)
    assert audio["daily_minutes_remaining"] is None


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


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ["daily", "monthly", "monthly-admission"])
async def test_postgres_audio_reads_backfill_legacy_usage_before_new_writes(
    audio_postgres, monkeypatch, reader: str
) -> None:
    from tldw_Server_API.app.core.Usage import audio_quota

    _client, pool = audio_postgres
    monkeypatch.setattr(audio_quota, "datetime", datetime)
    today = datetime.now(timezone.utc).date()
    await pool.execute(
        "INSERT INTO audio_usage_daily (user_id, day, minutes_used) VALUES ($1, $2, $3)",
        7, today, 2.5,
    )
    for _ in range(2):
        if reader == "monthly-admission":
            assert await audio_quota._monthly_minutes_exhausted(7, 3.0, 1.0) is True
        else:
            read = audio_quota.get_daily_minutes_used if reader == "daily" else audio_quota.get_monthly_minutes_used
            assert await read(7) == 2.5
