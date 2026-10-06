"""Audio quotas come from the user's limits.* values (spec 2 §4)."""

import pytest

from tldw_Server_API.app.core.Usage import audio_quota

pytestmark = pytest.mark.unit


@pytest.fixture()
def limits(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Quotas on, with per-key limits the test fills in."""
    values: dict = {}

    async def _user_quota(_user_id: int, key: str) -> object:
        """Serve the test's limit for this key."""
        return values.get(key)

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    monkeypatch.setattr(audio_quota, "user_quota", _user_quota)
    return values


async def test_limits_come_from_the_resolver(limits: dict) -> None:
    """Daily/monthly minutes and queued-job concurrency are the user's limits.* values."""
    limits.update({"limits.audio_daily_minutes": 45, "limits.transcription_minutes_per_month": 600, "limits.audio_concurrent_jobs": 3})
    assert await audio_quota.get_limits_for_user(1) == {
        "daily_minutes": 45, "monthly_minutes": 600, "concurrent_jobs": 3,
        "concurrent_streams": None, "max_file_size_mb": None,
    }


async def test_no_limits_set_means_unlimited(limits: dict) -> None:
    """A user with no limits.* values is unlimited even with quotas on."""
    allowed, remaining = await audio_quota.check_daily_minutes_allow(1, 500.0)
    assert allowed is True and remaining is None


async def test_monthly_minutes_block_when_exhausted(limits: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A month already at its limit refuses more minutes, checked before the daily ledger."""
    limits["limits.transcription_minutes_per_month"] = 60

    async def _sixty_minutes_used(_uid: str, _category: str) -> float:
        """3600 ledger seconds this month."""
        return 3600.0

    monkeypatch.setattr(audio_quota, "ledger_used_this_month", _sixty_minutes_used)
    assert await audio_quota.check_daily_minutes_allow(1, 1.0) == (False, 0.0)
    assert await audio_quota.consume_daily_minutes(1, 1.0) == (False, 0.0)


async def test_monthly_minutes_counter_error_fails_open_and_daily_path_still_runs(
    limits: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising monthly counter must not escape check_daily_minutes_allow/consume_daily_minutes;
    the daily path still runs, fail-open like check_usage (Qodo Q14)."""
    limits["limits.transcription_minutes_per_month"] = 60

    async def _raise(_uid: str, _category: str) -> float:
        """Simulate a monthly-counter lookup failure (e.g. a ledger/DB error)."""
        raise RuntimeError("ledger backend exploded")

    monkeypatch.setattr(audio_quota, "ledger_used_this_month", _raise)

    allowed, remaining = await audio_quota.check_daily_minutes_allow(1, 5.0)
    assert allowed is True and remaining is None

    added: list[object] = []

    class _FakeLedger:
        """A ledger stub recording add() calls, so the daily path's own write is observable."""

        async def add(self, entry: object) -> None:
            """Record the ledger write; the monthly counter failure must not have blocked it."""
            added.append(entry)

    async def _fake_get_daily_ledger():
        """Hand back the recording fake ledger."""
        return _FakeLedger()

    monkeypatch.setattr(audio_quota, "_get_daily_ledger", _fake_get_daily_ledger)

    allowed2, remaining2 = await audio_quota.consume_daily_minutes(1, 5.0)
    assert allowed2 is True and remaining2 is None
    assert len(added) == 1


async def test_synchronous_concurrency_is_unlimited(limits: dict) -> None:
    """can_start_job/can_start_stream admit every request (per-user sync concurrency is deferred)."""
    assert await audio_quota.can_start_job(1) == (True, "OK")
    assert await audio_quota.can_start_stream(1) == (True, "OK")
