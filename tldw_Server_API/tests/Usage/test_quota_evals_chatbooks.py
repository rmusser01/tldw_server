"""Per-user evaluation caps and chatbook job allowances (spec 2 §4)."""

import pytest

from tldw_Server_API.app.core.Chatbooks import chatbook_service
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookJobLimits
from tldw_Server_API.app.core.Chatbooks.exceptions import QuotaExceededError
from tldw_Server_API.app.core.Evaluations.user_rate_limiter import UserRateLimiter

pytestmark = pytest.mark.unit


def _limiter(tmp_path) -> UserRateLimiter:
    """A limiter on a throwaway SQLite DB with the schema in place."""
    return UserRateLimiter(db_path=str(tmp_path / "evals.db"))


async def test_daily_evaluation_cap_is_atomic_with_the_reservation(tmp_path) -> None:
    """The Nth+1 evaluation of the day is refused inside the same transaction that counts it."""
    limiter = _limiter(tmp_path)
    config = await limiter._get_user_config("7")
    for _ in range(2):
        ok, _meta = await limiter._reserve_request_usage("7", "/e", 0, 0.0, config, max_evaluations_per_day=2)
        assert ok is True
    ok, meta = await limiter._reserve_request_usage("7", "/e", 0, 0.0, config, max_evaluations_per_day=2)
    assert ok is False and meta["error"] == "Daily evaluation limit exceeded" and meta["limit"] == 2


async def test_daily_evaluation_token_cap(tmp_path) -> None:
    """A request whose tokens would pass the daily token cap is refused."""
    limiter = _limiter(tmp_path)
    config = await limiter._get_user_config("7")
    assert (await limiter._reserve_request_usage("7", "/e", 900, 0.0, config, max_tokens_per_day=1000))[0] is True
    ok, meta = await limiter._reserve_request_usage("7", "/e", 101, 0.0, config, max_tokens_per_day=1000)
    assert ok is False and meta["error"] == "Daily evaluation token limit exceeded"


class _Service(chatbook_service.ChatbookService):
    """A ChatbookService with the DB counters stubbed."""

    def __init__(self, exports: int, active: int) -> None:
        """Skip the real constructor; set only what admission reads."""
        self.user_id = "7"
        self.user_tier = "free"
        self.db = None
        self._exports = exports
        self._active = active

    def _count_operations_today_for_quota(self, operation_type: str) -> int:
        """Today's exports."""
        return self._exports

    def _count_active_jobs_for_quota(self) -> int:
        """Active jobs."""
        return self._active


def _quotas_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the test-mode flags that auto-disable chatbook quotas; switch quotas on."""
    for name in ("CHATBOOKS_DISABLE_QUOTAS", "TEST_MODE", "TESTING", "PYTEST_CURRENT_TEST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")


def test_chatbook_admission_uses_the_resolved_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exports/day and concurrency come from the passed limits; None is unlimited."""
    _quotas_live(monkeypatch)
    _Service(exports=100, active=100)._check_chatbook_job_admission("export", ChatbookJobLimits())
    with pytest.raises(QuotaExceededError):
        _Service(exports=3, active=0)._check_chatbook_job_admission("export", ChatbookJobLimits(exports_per_day=3))
    with pytest.raises(QuotaExceededError):
        _Service(exports=0, active=2)._check_chatbook_job_admission("export", ChatbookJobLimits(concurrent_jobs=2))


async def test_resolve_job_limits_reads_the_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """The service resolves its three chatbook keys for its user."""
    values = {"limits.chatbooks_exports_per_day": 4, "limits.chatbooks_concurrent_jobs": 1}

    async def _user_quota(_uid: int, key: str) -> object:
        """The test's values."""
        return values.get(key)

    monkeypatch.setattr(chatbook_service, "user_quota", _user_quota)
    service = _Service(exports=0, active=0)
    service.user_id_int = 7
    assert await service._resolve_job_limits() == ChatbookJobLimits(exports_per_day=4, imports_per_day=None, concurrent_jobs=1)
