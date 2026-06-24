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


@pytest.mark.asyncio
async def test_can_start_stream_and_finish_stream_via_rg_integration(monkeypatch):
    """
    Verify that can_start_stream/finish_stream use the ResourceGovernor path
    when it is available, by patching the internal governor accessor with a
    lightweight fake that enforces a simple max_concurrent=2 policy.
    """
    from tldw_Server_API.app.core.Usage import audio_quota

    class _FakeGov:
        def __init__(self):
            self.stream_counts = {}

        async def reserve(self, req, op_id=None):  # noqa: ARG002
            entity = req.entity
            cats = getattr(req, "categories", {}) or {}
            if "streams" not in cats:
                return type("Dec", (), {"allowed": True, "retry_after": 0})(), "h-ignore"
            count = self.stream_counts.get(entity, 0) + 1
            allowed = count <= 2
            if allowed:
                self.stream_counts[entity] = count
            dec = type("Dec", (), {"allowed": allowed, "retry_after": 0})()
            handle_id = f"h-{entity}-{count}" if allowed else None
            return dec, handle_id

        async def release(self, handle_id):  # noqa: ARG002
            # No-op for this test; audio_quota maintains its own handle registry.
            return None

        async def renew(self, handle_id, ttl_s):  # noqa: ARG002
            return None

    gov = _FakeGov()

    async def _fake_get_audio_rg():
        return gov

    # Force RG integration path regardless of env flags
    monkeypatch.setattr(audio_quota, "_get_audio_rg_governor", _fake_get_audio_rg)

    user_id = 123

    ok1, msg1 = await audio_quota.can_start_stream(user_id)
    assert ok1 is True
    assert msg1 in (None, "OK")

    ok2, msg2 = await audio_quota.can_start_stream(user_id)
    assert ok2 is True
    assert msg2 in (None, "OK")

    # Third concurrent stream should be denied by the fake RG policy.
    ok3, _ = await audio_quota.can_start_stream(user_id)
    assert ok3 is False

    # active_streams_count should reflect two active handles tracked by audio_quota.
    active = await audio_quota.active_streams_count(user_id)
    assert active == 2

    # Releasing one stream should reduce the active count.
    await audio_quota.finish_stream(user_id)
    active_after = await audio_quota.active_streams_count(user_id)
    assert active_after == 1


@pytest.mark.asyncio
async def test_can_start_stream_real_rg_enforces_limit_with_distinct_reservations(monkeypatch):
    from tldw_Server_API.app.core.Resource_Governance import MemoryResourceGovernor
    from tldw_Server_API.app.core.Usage import audio_quota

    audio_quota._reset_in_process_counters_for_tests()
    gov = MemoryResourceGovernor(
        policies={
            "audio.default": {
                "streams": {"max_concurrent": 2, "ttl_sec": 30},
                "scopes": ["user"],
            }
        }
    )

    async def _fake_get_audio_rg():
        return gov

    monkeypatch.setattr(audio_quota, "_get_audio_rg_governor", _fake_get_audio_rg)

    assert await audio_quota.can_start_stream(211) == (True, "OK")
    assert await audio_quota.can_start_stream(211) == (True, "OK")

    allowed, reason = await audio_quota.can_start_stream(211)
    assert allowed is False
    assert "Concurrent streams" in reason


@pytest.mark.asyncio
async def test_can_start_job_real_rg_enforces_limit_with_distinct_reservations(monkeypatch):
    from tldw_Server_API.app.core.Resource_Governance import MemoryResourceGovernor
    from tldw_Server_API.app.core.Usage import audio_quota

    audio_quota._reset_in_process_counters_for_tests()
    gov = MemoryResourceGovernor(
        policies={
            "audio.default": {
                "jobs": {"max_concurrent": 1, "ttl_sec": 30},
                "scopes": ["user"],
            }
        }
    )

    async def _fake_get_audio_rg():
        return gov

    monkeypatch.setattr(audio_quota, "_get_audio_rg_governor", _fake_get_audio_rg)

    assert await audio_quota.can_start_job(212) == (True, "OK")

    allowed, reason = await audio_quota.can_start_job(212)
    assert allowed is False
    assert "Concurrent job" in reason


@pytest.mark.asyncio
async def test_can_start_job_fails_open_when_rg_unavailable(monkeypatch):
    """
    Legacy concurrency counters are retired. If RG is enabled but the governor
    accessor returns None, can_start_job should fail open to avoid blocking jobs.
    """
    from tldw_Server_API.app.core.Usage import audio_quota

    monkeypatch.setenv("RG_ENABLED", "1")

    # Force RG path to be unavailable
    async def _rg_unavailable():
        return None

    monkeypatch.setattr(audio_quota, "_get_audio_rg_governor", _rg_unavailable)

    user_id = 99
    ok1, msg1 = await audio_quota.can_start_job(user_id)
    assert ok1 is True
    assert msg1 in (None, "OK")
