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


class _DenyingGovernor:
    """A governor that fails the test if it is consulted."""

    async def reserve(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("the governor must not be consulted when quotas are off")

    async def release(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("no lease was taken, so none may be released")


async def _denying_governor() -> _DenyingGovernor:
    return _DenyingGovernor()


async def test_limits_are_unlimited_when_quotas_off(quotas_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _no_tier_lookup(_user_id: int) -> str:
        raise AssertionError("the tier lookup must not run when quotas are off")

    monkeypatch.setattr(audio_quota, "get_user_tier", _no_tier_lookup)
    limits = await audio_quota.get_limits_for_user(1)
    assert limits == {"daily_minutes": None, "concurrent_streams": None, "concurrent_jobs": None, "max_file_size_mb": None}


async def test_daily_minutes_allowed_past_the_old_free_tier(quotas_off: None) -> None:
    allowed, remaining = await audio_quota.check_daily_minutes_allow(1, 31.0)
    assert allowed is True
    assert remaining is None


async def test_concurrency_unlimited_when_quotas_off(quotas_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_quota, "_get_audio_rg_governor", _denying_governor)
    assert await audio_quota.can_start_job(1) == (True, "OK")
    assert await audio_quota.can_start_stream(1) == (True, "OK")


async def test_finish_without_lease_is_a_noop(quotas_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_quota, "_get_audio_rg_governor", _denying_governor)
    await audio_quota.can_start_stream(5)
    await audio_quota.finish_stream(5)
    await audio_quota.can_start_job(5)
    await audio_quota.finish_job(5)


async def test_quotas_on_keeps_the_tier_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")

    async def _free(_user_id: int) -> str:
        return "free"

    async def _no_overrides(_user_id: int) -> dict:
        return {}

    monkeypatch.setattr(audio_quota, "get_user_tier", _free)
    monkeypatch.setattr(audio_quota, "_get_user_override_limits", _no_overrides)
    limits = await audio_quota.get_limits_for_user(1)
    assert limits["daily_minutes"] == audio_quota.TIER_LIMITS["free"]["daily_minutes"]


def test_upload_cap_falls_back_to_the_media_processing_cap() -> None:
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_transcriptions

    assert audio_transcriptions._max_upload_bytes({"max_file_size_mb": None}) == Audio_Files.MAX_FILE_SIZE
    assert audio_transcriptions._max_upload_bytes({}) == Audio_Files.MAX_FILE_SIZE
    assert audio_transcriptions._max_upload_bytes({"max_file_size_mb": "junk"}) == Audio_Files.MAX_FILE_SIZE
    assert audio_transcriptions._max_upload_bytes({"max_file_size_mb": 25}) == 25 * 1024 * 1024
