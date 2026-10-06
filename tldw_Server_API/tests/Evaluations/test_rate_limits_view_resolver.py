"""The evaluations limits view shows the enforced limits.* daily caps (spec 2 §9)."""

import pytest

from tldw_Server_API.app.core.Evaluations import user_rate_limiter
from tldw_Server_API.app.core.Evaluations.user_rate_limiter import UserRateLimiter

pytestmark = pytest.mark.unit


@pytest.fixture()
def caps(monkeypatch: pytest.MonkeyPatch) -> dict:
    """limits.* daily caps the test controls."""
    values: dict = {}

    async def _user_quota(user_id, key: str):
        """The test's cap for the key."""
        return values.get(key)

    monkeypatch.setattr(user_rate_limiter, "user_quota", _user_quota)
    return values


async def test_summary_unlimited_when_no_cap_set(caps: dict, tmp_path) -> None:
    """No limits.* daily caps: limits and remaining are None, not the tier's 100/day."""
    limiter = UserRateLimiter(db_path=str(tmp_path / "evals.db"))
    summary = await limiter.get_usage_summary("7")
    assert summary["limits"]["daily"]["evaluations"] is None
    assert summary["limits"]["daily"]["tokens"] is None
    assert summary["remaining"]["daily_evaluations"] is None
    assert summary["remaining"]["daily_tokens"] is None


async def test_summary_reports_the_resolved_cap(caps: dict, tmp_path) -> None:
    """A limits.evaluations_per_day of 3 is reported with its remaining count."""
    caps["limits.evaluations_per_day"] = 3
    caps["limits.evaluation_tokens_per_day"] = 0
    limiter = UserRateLimiter(db_path=str(tmp_path / "evals.db"))
    summary = await limiter.get_usage_summary("7")
    assert summary["limits"]["daily"]["evaluations"] == 3
    assert summary["remaining"]["daily_evaluations"] == 3
    assert summary["limits"]["daily"]["tokens"] == 0
    assert summary["remaining"]["daily_tokens"] == 0


async def _headers_for(tmp_path) -> dict:
    """Headers `_apply_rate_limit_headers` sets for user 7 on a fresh limiter."""
    from fastapi import Response

    from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth import _apply_rate_limit_headers

    response = Response()
    limiter = UserRateLimiter(db_path=str(tmp_path / "evals.db"))
    await _apply_rate_limit_headers(limiter, "7", response)
    return dict(response.headers)


async def test_daily_headers_present_when_a_cap_is_set(caps: dict, tmp_path) -> None:
    """A set cap is advertised in the three daily headers."""
    caps["limits.evaluations_per_day"] = 5
    caps["limits.evaluation_tokens_per_day"] = 1000
    headers = await _headers_for(tmp_path)
    assert headers["x-ratelimit-daily-limit"] == "5"
    assert headers["x-ratelimit-daily-remaining"] == "5"
    assert headers["x-ratelimit-tokens-remaining"] == "1000"


async def test_daily_headers_absent_when_unlimited(caps: dict, tmp_path) -> None:
    """No cap set: the three daily headers are left out."""
    headers = await _headers_for(tmp_path)
    assert "x-ratelimit-daily-limit" not in headers
    assert "x-ratelimit-daily-remaining" not in headers
    assert "x-ratelimit-tokens-remaining" not in headers
