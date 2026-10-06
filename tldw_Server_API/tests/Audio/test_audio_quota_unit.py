import asyncio
from unittest.mock import AsyncMock

import pytest


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)


class _FailingAudioQuotaPool:
    pool = None

    def __init__(self, *, fail_fetchone: bool = False, fail_fetch: bool = False, fail_set_execute: bool = False) -> None:
        self.fail_fetchone = fail_fetchone
        self.fail_fetch = fail_fetch
        self.fail_set_execute = fail_set_execute
        self.execute_calls = 0

    async def execute(self, *args, **kwargs) -> None:
        self.execute_calls += 1
        if self.fail_set_execute and self.execute_calls > 2:
            raise RuntimeError("audio quota write failed at /private/audio-quota.db")

    async def fetchone(self, *args, **kwargs):
        if self.fail_fetchone:
            raise RuntimeError("audio tier read failed at /private/audio-quota.db")
        return None

    async def fetch(self, *args, **kwargs):
        if self.fail_fetch:
            raise RuntimeError("audio minutes read failed at /private/audio-quota.db")
        return []


class _AlwaysFailingAudioQuotaPool:
    pool = None

    async def execute(self, *args, **kwargs) -> None:
        raise RuntimeError("audio table ensure failed at /private/audio-quota.db")


@pytest.mark.asyncio
async def test_daily_ledger_init_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    class _FailingLedger:
        async def initialize(self) -> None:
            raise RuntimeError("audio ledger init failed at /private/audio-ledger.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_quota, "_daily_ledger", None)
    monkeypatch.setattr(audio_quota, "ResourceDailyLedger", _FailingLedger)
    monkeypatch.setattr(audio_quota, "LedgerEntry", object())
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    ledger = await audio_quota._get_daily_ledger()

    assert ledger is None
    assert logger_stub.debugs == ["Audio quotas ResourceDailyLedger init failed; continuing without ledger"]
    assert "audio ledger init failed" not in str(logger_stub.debugs)
    assert "/private/audio-ledger.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_add_daily_minutes_ledger_add_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    class _FailingLedger:
        async def add(self, entry) -> None:  # noqa: ARG002
            raise RuntimeError("audio ledger add failed at /private/audio-ledger.db")

    async def _fake_get_daily_ledger():
        return _FailingLedger()

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_quota, "_get_daily_ledger", _fake_get_daily_ledger)
    monkeypatch.setattr(audio_quota, "LedgerEntry", lambda **kwargs: kwargs)
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    await audio_quota.add_daily_minutes(user_id=7, minutes=1.25)

    assert logger_stub.debugs == ["Audio quotas ResourceDailyLedger add failed; shadow-only"]
    assert "audio ledger add failed" not in str(logger_stub.debugs)
    assert "/private/audio-ledger.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_ensure_tables_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    await audio_quota._ensure_tables(_AlwaysFailingAudioQuotaPool())

    assert logger_stub.debugs == ["audio_usage_daily ensure failed"]
    assert "audio table ensure failed" not in str(logger_stub.debugs)
    assert "/private/audio-quota.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_get_user_tier_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    async def _fake_get_db_pool():
        return _FailingAudioQuotaPool(fail_fetchone=True)

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_quota, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    tier = await audio_quota.get_user_tier(123)

    assert tier == "free"
    assert logger_stub.debugs == ["get_user_tier failed"]
    assert "audio tier read failed" not in str(logger_stub.debugs)
    assert "/private/audio-quota.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_set_user_tier_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    async def _fake_get_db_pool():
        return _FailingAudioQuotaPool(fail_set_execute=True)

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_quota, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    with pytest.raises(RuntimeError):
        await audio_quota.set_user_tier(123, "premium")

    assert logger_stub.debugs == ["set_user_tier failed"]
    assert "123" not in str(logger_stub.debugs)
    assert "premium" not in str(logger_stub.debugs)
    assert "audio quota write failed" not in str(logger_stub.debugs)
    assert "/private/audio-quota.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_get_daily_minutes_used_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    async def _failing_ledger_used_today(entity_value: str, category: str) -> float:  # noqa: ARG001
        raise RuntimeError("audio minutes read failed at /private/audio-quota.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_quota, "ledger_used_today", _failing_ledger_used_today)
    monkeypatch.setattr(audio_quota, "_get_daily_ledger", AsyncMock(return_value=None))
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    minutes = await audio_quota.get_daily_minutes_used(123)

    assert minutes == 0.0
    assert logger_stub.debugs == ["get_daily_minutes_used failed"]
    assert "audio minutes read failed" not in str(logger_stub.debugs)
    assert "/private/audio-quota.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_ledger_remaining_minutes_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    class _FailingLedger:
        async def remaining_for_day(self, **kwargs):  # noqa: ARG002
            raise RuntimeError("audio remaining failed at /private/audio-ledger.db")

    async def _fake_get_daily_ledger():
        return _FailingLedger()

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_quota, "_get_daily_ledger", _fake_get_daily_ledger)
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    remaining = await audio_quota._ledger_remaining_minutes(user_id=123, daily_limit_minutes=30.0)

    assert remaining is None
    assert logger_stub.debugs == ["Audio quotas ledger remaining check failed; fallback to legacy"]
    assert "audio remaining failed" not in str(logger_stub.debugs)
    assert "/private/audio-ledger.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_increment_jobs_started_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    async def _fake_get_db_pool():
        return _FailingAudioQuotaPool(fail_set_execute=True)

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_quota, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    await audio_quota.increment_jobs_started(123)

    assert logger_stub.debugs == ["increment_jobs_started failed"]
    assert "audio quota write failed" not in str(logger_stub.debugs)
    assert "/private/audio-quota.db" not in str(logger_stub.debugs)


def test_stream_ttl_invalid_env_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    logger_stub = _LoggerStub()
    monkeypatch.setenv("AUDIO_STREAM_TTL_SECONDS", "bad-stream-ttl-/private/audio")
    monkeypatch.setattr(audio_quota, "logger", logger_stub)
    audio_quota._get_stream_ttl_seconds.cache_clear()

    ttl = audio_quota._get_stream_ttl_seconds()

    assert ttl >= 30
    assert logger_stub.debugs == ["Audio stream TTL: invalid AUDIO_STREAM_TTL_SECONDS"]
    assert "bad-stream-ttl" not in str(logger_stub.debugs)
    assert "/private/audio" not in str(logger_stub.debugs)
    audio_quota._get_stream_ttl_seconds.cache_clear()


def test_job_ttl_invalid_env_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    logger_stub = _LoggerStub()
    monkeypatch.setenv("AUDIO_JOB_TTL_SECONDS", "bad-job-ttl-/private/audio")
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    ttl = audio_quota._get_job_ttl_seconds()

    assert ttl >= 30
    assert logger_stub.debugs == ["Audio job TTL: invalid AUDIO_JOB_TTL_SECONDS"]
    assert "bad-job-ttl" not in str(logger_stub.debugs)
    assert "/private/audio" not in str(logger_stub.debugs)


def test_audio_tier_limits_json_parse_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    logger_stub = _LoggerStub()
    base = {"free": {"daily_minutes": 30.0, "concurrent_streams": 1}}
    monkeypatch.setenv("AUDIO_TIER_LIMITS_JSON", "bad-json-/private/audio-limits.json")
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    merged = audio_quota._apply_tier_overrides_from_config(base)

    assert merged["free"]["daily_minutes"] == 30.0
    assert logger_stub.debugs == ["AUDIO_TIER_LIMITS_JSON parse failed"]
    assert "bad-json" not in str(logger_stub.debugs)
    assert "/private/audio-limits.json" not in str(logger_stub.debugs)


def test_audio_tier_config_override_value_log_is_sanitized(monkeypatch):
    import configparser

    from tldw_Server_API.app.core import config as config_module
    from tldw_Server_API.app.core.Usage import audio_quota

    cfg = configparser.ConfigParser()
    cfg.add_section("Audio-Quota")
    cfg.set("Audio-Quota", "free_daily_minutes", "not-a-number-/private/config.txt")

    logger_stub = _LoggerStub()
    base = {"free": {"daily_minutes": 30.0, "concurrent_streams": 1}}
    monkeypatch.delenv("AUDIO_TIER_LIMITS_JSON", raising=False)
    monkeypatch.setattr(config_module, "load_comprehensive_config", lambda: cfg)
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    merged = audio_quota._apply_tier_overrides_from_config(base)

    assert merged["free"]["daily_minutes"] == 30.0
    assert logger_stub.debugs == ["Audio-Quota override parse failed"]
    assert "not-a-number" not in str(logger_stub.debugs)
    assert "/private/config.txt" not in str(logger_stub.debugs)


def test_audio_tier_config_override_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core import config as config_module
    from tldw_Server_API.app.core.Usage import audio_quota

    def _failing_load_config():
        raise RuntimeError("audio quota config failed at /private/config.txt")

    logger_stub = _LoggerStub()
    base = {"free": {"daily_minutes": 30.0, "concurrent_streams": 1}}
    monkeypatch.delenv("AUDIO_TIER_LIMITS_JSON", raising=False)
    monkeypatch.setattr(config_module, "load_comprehensive_config", _failing_load_config)
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    merged = audio_quota._apply_tier_overrides_from_config(base)

    assert merged["free"]["daily_minutes"] == 30.0
    assert logger_stub.debugs == ["Audio-Quota config overrides failed"]
    assert "audio quota config failed" not in str(logger_stub.debugs)
    assert "/private/config.txt" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_add_daily_minutes_shadow_path_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    async def _failing_get_daily_ledger():
        raise RuntimeError("audio ledger shadow path failed at /private/audio-ledger.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_quota, "_get_daily_ledger", _failing_get_daily_ledger)
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    await audio_quota.add_daily_minutes(user_id=123, minutes=1.0)

    assert logger_stub.debugs == ["Audio quotas: ResourceDailyLedger shadow path failed; ignoring"]
    assert "audio ledger shadow path failed" not in str(logger_stub.debugs)
    assert "/private/audio-ledger.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_get_daily_ledger_backfill_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    class _Ledger:
        async def initialize(self) -> None:
            return None

    async def _failing_backfill(ledger):  # noqa: ARG001
        raise RuntimeError("legacy backfill failed at /private/audio-legacy.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_quota, "_daily_ledger", None)
    monkeypatch.setattr(audio_quota, "ResourceDailyLedger", _Ledger)
    monkeypatch.setattr(audio_quota, "LedgerEntry", object())
    monkeypatch.setattr(audio_quota, "_backfill_audio_usage_daily_to_ledger", _failing_backfill)
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    ledger = await audio_quota._get_daily_ledger()

    assert ledger is not None
    assert logger_stub.debugs == ["Audio quotas: legacy audio_usage_daily backfill failed; continuing without backfill"]
    assert "legacy backfill failed" not in str(logger_stub.debugs)
    assert "/private/audio-legacy.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_backfill_postgres_query_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    class _FailingPostgresPool:
        pool = object()

        async def fetch(self, *args, **kwargs):  # noqa: ARG002
            raise RuntimeError("postgres backfill query failed at /private/audio-pg.db")

    async def _fake_get_db_pool():
        return _FailingPostgresPool()

    async def _noop_ensure_tables(pool):  # noqa: ARG001
        return None

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_quota, "_audio_minutes_legacy_backfill_done", False)
    monkeypatch.setattr(audio_quota, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(audio_quota, "_ensure_tables", _noop_ensure_tables)
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    await audio_quota._backfill_audio_usage_daily_to_ledger(object())

    assert logger_stub.debugs == ["Audio quotas: legacy backfill query (Postgres) failed"]
    assert "postgres backfill query failed" not in str(logger_stub.debugs)
    assert "/private/audio-pg.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_backfill_sqlite_query_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    class _FailingSQLitePool:
        pool = None

        async def fetch(self, *args, **kwargs):  # noqa: ARG002
            raise RuntimeError("sqlite backfill query failed at /private/audio-sqlite.db")

    async def _fake_get_db_pool():
        return _FailingSQLitePool()

    async def _noop_ensure_tables(pool):  # noqa: ARG001
        return None

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_quota, "_audio_minutes_legacy_backfill_done", False)
    monkeypatch.setattr(audio_quota, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(audio_quota, "_ensure_tables", _noop_ensure_tables)
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    await audio_quota._backfill_audio_usage_daily_to_ledger(object())

    assert logger_stub.debugs == ["Audio quotas: legacy backfill query (SQLite) failed"]
    assert "sqlite backfill query failed" not in str(logger_stub.debugs)
    assert "/private/audio-sqlite.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_backfill_ledger_add_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    class _SQLitePool:
        pool = None

        async def fetch(self, *args, **kwargs):  # noqa: ARG002
            return [(123, 1.0)]

    class _FailingLedger:
        async def add(self, entry) -> None:  # noqa: ARG002
            raise RuntimeError("ledger backfill add failed at /private/audio-ledger.db")

    async def _fake_get_db_pool():
        return _SQLitePool()

    async def _noop_ensure_tables(pool):  # noqa: ARG001
        return None

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_quota, "_audio_minutes_legacy_backfill_done", False)
    monkeypatch.setattr(audio_quota, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(audio_quota, "_ensure_tables", _noop_ensure_tables)
    monkeypatch.setattr(audio_quota, "LedgerEntry", lambda **kwargs: kwargs)
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    await audio_quota._backfill_audio_usage_daily_to_ledger(_FailingLedger())

    assert logger_stub.debugs == ["Audio quotas: ResourceDailyLedger legacy backfill add failed"]
    assert "ledger backfill add failed" not in str(logger_stub.debugs)
    assert "/private/audio-ledger.db" not in str(logger_stub.debugs)
    assert "123" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_backfill_outer_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import audio_quota

    async def _failing_get_db_pool():
        raise RuntimeError("legacy backfill pool failed at /private/audio-pool.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(audio_quota, "_audio_minutes_legacy_backfill_done", False)
    monkeypatch.setattr(audio_quota, "get_db_pool", _failing_get_db_pool)
    monkeypatch.setattr(audio_quota, "logger", logger_stub)

    await audio_quota._backfill_audio_usage_daily_to_ledger(object())

    assert logger_stub.debugs == [
        "Audio quotas: legacy audio_usage_daily backfill to ResourceDailyLedger failed; continuing without backfill"
    ]
    assert "legacy backfill pool failed" not in str(logger_stub.debugs)
    assert "/private/audio-pool.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_user_tier_default_and_set_roundtrip():
    from tldw_Server_API.app.core.Usage.audio_quota import get_user_tier, set_user_tier
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    uid = 777777
    pool = await get_db_pool()
    # Ensure deterministic state for this uid
    await set_user_tier(uid, "free")
    await pool.execute("DELETE FROM audio_user_tiers WHERE user_id = ?", uid)

    # default is free when no row exists
    assert (await get_user_tier(uid)) == "free"
    await set_user_tier(uid, "premium")
    assert (await get_user_tier(uid)) == "premium"
