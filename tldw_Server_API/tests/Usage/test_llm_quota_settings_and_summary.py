"""Startup warnings for settings that defeat the LLM quotas, and the profile's LLM block (spec §6, §9)."""

from types import SimpleNamespace

import pytest
from fastapi.encoders import jsonable_encoder

from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import UserProfileQuotas
from tldw_Server_API.app.core.Usage import quota_checks
from tldw_Server_API.app.services import startup_auth_runtime

pytestmark = pytest.mark.unit


@pytest.fixture
def quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas on, recording on, a 180-day retention, and the warn-once flag cleared."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    monkeypatch.setenv("LLM_USAGE_ENABLED", "1")
    monkeypatch.setattr(quota_checks, "_LLM_QUOTA_SETTINGS_WARNED", False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.settings.get_settings",
        lambda: SimpleNamespace(LLM_USAGE_LOG_RETENTION_DAYS=180),
    )


def test_sound_settings_warn_nothing(quotas_on: None) -> None:
    """Quotas on with recording on and a long retention: no warning."""
    assert quota_checks.warn_if_llm_quotas_defeated() == []


def test_recording_off_is_reported_once(quotas_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """With LLM_USAGE_ENABLED off the LLM quotas never refuse; the warning names the setting, once per process."""
    monkeypatch.setenv("LLM_USAGE_ENABLED", "0")
    problems = quota_checks.warn_if_llm_quotas_defeated()
    assert len(problems) == 1 and "LLM_USAGE_ENABLED" in problems[0]
    assert quota_checks.warn_if_llm_quotas_defeated() == []


def test_a_short_retention_is_reported(quotas_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Under 31 days of retention, the monthly counters lose rows they need."""
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.settings.get_settings",
        lambda: SimpleNamespace(LLM_USAGE_LOG_RETENTION_DAYS=7),
    )
    problems = quota_checks.warn_if_llm_quotas_defeated()
    assert len(problems) == 1 and "LLM_USAGE_LOG_RETENTION_DAYS" in problems[0] and "31" in problems[0]


def test_nothing_is_reported_with_quotas_off(quotas_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """With quotas off nothing is enforced, so nothing is defeated."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "0")
    monkeypatch.setenv("LLM_USAGE_ENABLED", "0")
    assert quota_checks.warn_if_llm_quotas_defeated() == []


async def test_startup_runs_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The auth runtime startup step calls the check."""
    calls: list[str] = []
    monkeypatch.setattr(quota_checks, "warn_if_llm_quotas_defeated", lambda: calls.append("checked") or [])
    startup_auth_runtime._warn_if_llm_quotas_defeated()
    assert calls == ["checked"]


async def test_the_summary_reports_usage_and_set_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Usage is always reported; a limit appears only when it is set."""

    async def _user_quota(_uid: object, key: str) -> float | None:
        return 2.5 if key == "limits.llm_cost_per_day_usd" else None

    async def _value(value: float):
        return value

    monkeypatch.setattr(quota_checks, "user_quota", _user_quota)
    monkeypatch.setattr(quota_checks, "llm_tokens_this_month", lambda _uid: _value(1234.0))
    monkeypatch.setattr(quota_checks, "llm_cost_today", lambda _uid: _value(0.123456))
    monkeypatch.setattr(quota_checks, "llm_cost_this_month", lambda _uid: _value(4.5))
    summary = await quota_checks.llm_quota_summary(7)
    assert summary == {
        "tokens_month_used": 1234,
        "tokens_month_limit": None,
        "cost_day_usd_used": 0.1235,
        "cost_day_usd_limit": 2.5,
        "cost_month_usd_used": 4.5,
        "cost_month_usd_limit": None,
    }
    # Both profile endpoints use response_model_exclude_none=True, which FastAPI applies with
    # jsonable_encoder: it drops None inside nested dicts, where model_dump(exclude_none=True) does not.
    dumped = jsonable_encoder(UserProfileQuotas(storage_used_mb=0.0, llm=summary), exclude_none=True)
    assert dumped["llm"] == {
        "tokens_month_used": 1234,
        "cost_day_usd_used": 0.1235,
        "cost_day_usd_limit": 2.5,
        "cost_month_usd_used": 4.5,
    }
