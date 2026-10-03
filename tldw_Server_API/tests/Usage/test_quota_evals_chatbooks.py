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


async def test_daily_evaluation_cap_of_zero_refuses_the_first_evaluation(tmp_path) -> None:
    """max_evaluations_per_day == 0 must block, not mean unlimited (spec 2 review B2)."""
    limiter = _limiter(tmp_path)
    config = await limiter._get_user_config("7")
    ok, meta = await limiter._reserve_request_usage("7", "/e", 0, 0.0, config, max_evaluations_per_day=0)
    assert ok is False and meta["error"] == "Daily evaluation limit exceeded" and meta["limit"] == 0


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


def test_chatbooks_exports_per_day_of_zero_refuses_the_first_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """exports_per_day == 0 must block, not mean unlimited (spec 2 review B2)."""
    _quotas_live(monkeypatch)
    with pytest.raises(QuotaExceededError):
        _Service(exports=0, active=0)._check_chatbook_job_admission("export", ChatbookJobLimits(exports_per_day=0))


def test_chatbooks_concurrent_jobs_of_zero_refuses_the_first_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """concurrent_jobs == 0 must block, not mean unlimited (spec 2 review B2)."""
    _quotas_live(monkeypatch)
    with pytest.raises(QuotaExceededError):
        _Service(exports=0, active=0)._check_chatbook_job_admission("export", ChatbookJobLimits(concurrent_jobs=0))


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


def test_evaluations_endpoint_daily_cap_fires_for_the_authenticated_user(tmp_path, monkeypatch) -> None:
    """A non-numeric credential must not blind the per-user daily evaluation cap (spec 2 §4).

    `verify_api_key` returns the raw API key/JWT string, not a user id. The limiter must be
    keyed by `current_user.id` (the authenticated user), not by that credential, or
    `as_quota_user_id` never resolves and the daily cap never fires for real traffic.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.Usage import quota_resolver

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    monkeypatch.setenv("RG_ENABLED", "0")

    fresh_limiter = UserRateLimiter(db_path=str(tmp_path / "endpoint_evals.db"))
    monkeypatch.setattr(evaluations_unified, "get_user_rate_limiter_for_user", lambda _uid: fresh_limiter)

    class _FakeEvalService:
        """Records calls instead of running a real proposition evaluation."""

        def __init__(self) -> None:
            """Start with no recorded calls."""
            self.calls: list[dict] = []

        async def evaluate_propositions(self, **kwargs) -> dict:
            """Record the call; return a minimal well-formed result."""
            self.calls.append(kwargs)
            return {
                "results": {"metrics": {}, "counts": {}, "details": {}},
                "evaluation_id": "fake-eval",
                "evaluation_time": 0.0,
            }

    fake_service = _FakeEvalService()
    monkeypatch.setattr(evaluations_unified, "get_unified_evaluation_service_for_user", lambda _uid: fake_service)

    seen_uids: list[object] = []

    async def _fake_user_quota(uid: object, key: str) -> object:
        """Only a real (numeric) user id gets the daily cap of 1; a credential string must not."""
        seen_uids.append(uid)
        if key == "limits.evaluations_per_day" and isinstance(uid, int):
            return 1
        return None

    monkeypatch.setattr(quota_resolver, "user_quota", _fake_user_quota)

    app = FastAPI()
    app.include_router(evaluations_unified.router, prefix="/api/v1")
    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    payload = {
        "extracted": ["Alice founded Acme Corp in 2020"],
        "reference": ["Alice founded Acme Corp in 2020"],
        "method": "jaccard",
        "threshold": 0.5,
    }

    with TestClient(app) as client:
        first = client.post("/api/v1/evaluations/propositions", json=payload, headers=headers)
        second = client.post("/api/v1/evaluations/propositions", json=payload, headers=headers)

    assert first.status_code == 200, first.text
    assert second.status_code == 429, second.text
    assert len(fake_service.calls) == 1
    assert any(isinstance(uid, int) for uid in seen_uids), seen_uids
