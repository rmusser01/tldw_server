"""Audio quotas are unlimited when usage quotas are off (spec 2)."""

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files
from tldw_Server_API.app.core.Usage import audio_quota

pytestmark = pytest.mark.unit


@pytest.fixture()
def quotas_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stock install: neither switch spelling is set."""
    monkeypatch.delenv("USAGE_QUOTAS_ENABLED", raising=False)
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: None)


async def test_limits_are_unlimited_when_quotas_off(quotas_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """With quotas off, get_limits_for_user returns all-None limits without ever looking up the tier."""

    async def _no_tier_lookup(_user_id: int) -> str:
        """A tier-lookup stand-in that fails the test if it is ever called."""
        raise AssertionError("the tier lookup must not run when quotas are off")

    monkeypatch.setattr(audio_quota, "get_user_tier", _no_tier_lookup)
    limits = await audio_quota.get_limits_for_user(1)
    assert limits == {
        "daily_minutes": None,
        "monthly_minutes": None,
        "concurrent_streams": None,
        "concurrent_jobs": None,
        "max_file_size_mb": None,
    }


async def test_daily_minutes_allowed_past_the_old_free_tier(quotas_off: None) -> None:
    """With quotas off, daily minutes usage past the old free-tier cap is still allowed with no remaining limit."""
    allowed, remaining = await audio_quota.check_daily_minutes_allow(1, 31.0)
    assert allowed is True
    assert remaining is None


async def test_concurrency_hooks_admit_and_finish_is_a_noop(quotas_off: None) -> None:
    """With quotas off, the can_start hooks admit and the finish hooks do nothing and do not raise."""
    assert await audio_quota.can_start_job(1) == (True, "OK")
    assert await audio_quota.can_start_stream(1) == (True, "OK")
    assert await audio_quota.finish_job(1) is None
    assert await audio_quota.finish_stream(1) is None


async def test_quotas_on_reads_daily_minutes_from_the_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """With quotas on, get_limits_for_user's daily_minutes is the user's limits.audio_daily_minutes value (spec 2)."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")

    async def _user_quota(_user_id: int, key: str) -> object:
        """Serve a daily-minutes override; every other key is unset (unlimited)."""
        return 45 if key == "limits.audio_daily_minutes" else None

    monkeypatch.setattr(audio_quota, "user_quota", _user_quota)
    limits = await audio_quota.get_limits_for_user(1)
    assert limits["daily_minutes"] == 45


def test_upload_cap_falls_back_to_the_media_processing_cap() -> None:
    """With no max_file_size_mb override, the upload cap falls back to the media-processing default for every missing or invalid value."""
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_transcriptions

    assert audio_transcriptions._max_upload_bytes({"max_file_size_mb": None}) == Audio_Files.MAX_FILE_SIZE
    assert audio_transcriptions._max_upload_bytes({}) == Audio_Files.MAX_FILE_SIZE
    assert audio_transcriptions._max_upload_bytes({"max_file_size_mb": "junk"}) == Audio_Files.MAX_FILE_SIZE
    assert audio_transcriptions._max_upload_bytes({"max_file_size_mb": 25}) == 25 * 1024 * 1024
