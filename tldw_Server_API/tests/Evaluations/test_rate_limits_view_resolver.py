"""The evaluations limits view shows the enforced limits.* daily caps (spec 2 §9)."""

import pytest

from tldw_Server_API.app.core.Evaluations.user_rate_limiter import UserRateLimiter
from tldw_Server_API.app.core.Usage import quota_resolver

pytestmark = pytest.mark.unit


@pytest.fixture()
def caps(monkeypatch: pytest.MonkeyPatch) -> dict:
    """limits.* daily caps the test controls."""
    values: dict = {}

    async def _user_quota(user_id, key: str):
        """The test's cap for the key."""
        return values.get(key)

    monkeypatch.setattr(quota_resolver, "user_quota", _user_quota)
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
