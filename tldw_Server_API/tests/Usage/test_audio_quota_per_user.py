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


async def test_synchronous_concurrency_is_unlimited(limits: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """can_start_job/can_start_stream no longer reserve RG leases (per-user sync concurrency is deferred)."""
    fetched: list[str] = []

    async def _recording_governor() -> None:
        """Record any governor fetch."""
        fetched.append("governor")

    monkeypatch.setattr(audio_quota, "_get_audio_rg_governor", _recording_governor)
    assert await audio_quota.can_start_job(1) == (True, "OK")
    assert await audio_quota.can_start_stream(1) == (True, "OK")
    assert fetched == []
