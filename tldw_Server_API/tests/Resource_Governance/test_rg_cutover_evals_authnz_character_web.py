import asyncio
import warnings

import pytest

from tldw_Server_API.app.core.Character_Chat import character_rate_limiter as char_rl
from tldw_Server_API.app.core.Evaluations import user_rate_limiter as evals_rl
from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as web_rl


class _FakeDecision:
    def __init__(self, allowed: bool, retry_after: int | None = None):
        self.allowed = allowed
        self.retry_after = retry_after
        self.details = {}


class _FakeGovernor:
    def __init__(self, allowed: bool = True, retry_after: int | None = None):
        self.allowed = allowed
        self.retry_after = retry_after
        self.reserved = []
        self.commits = []

    async def reserve(self, req, op_id=None):
        self.reserved.append((req.entity, req.categories, op_id, req.tags))
        return _FakeDecision(self.allowed, self.retry_after), "handle-1"

    async def commit(self, handle_id, actuals=None, op_id=None):
        self.commits.append((handle_id, actuals, op_id))


@pytest.mark.asyncio
async def test_evaluations_rg_denies(monkeypatch):
    monkeypatch.setenv("RG_ENABLED", "1")
    fake = _FakeGovernor(allowed=False, retry_after=5)
    monkeypatch.setattr(evals_rl, "_rg_evals_governor", fake)
    monkeypatch.setattr(evals_rl, "_rg_evals_loader", None)

    limiter = evals_rl.UserRateLimiter()

    allowed, meta = await limiter.check_rate_limit(
        user_id="user-123",
        endpoint="/api/v1/evaluations",
    )

    assert allowed is False
    assert meta.get("retry_after") == 5
    assert meta.get("policy_id") == "evals.free"
    assert fake.reserved
    entity, categories, _op_id, tags = fake.reserved[-1]
    assert entity == "user:user-123"
    assert categories == {"evaluations": {"units": 1}}
    assert tags.get("module") == "evaluations"


@pytest.mark.asyncio
async def test_evaluations_rg_allows_bypasses_legacy(monkeypatch):
    """
    Phase 2: When RG returns allow, the evaluations limiter should allow
    the request. No legacy minute/daily checks exist to override.
    """
    monkeypatch.setenv("RG_ENABLED", "1")
    fake = _FakeGovernor(allowed=True, retry_after=None)
    monkeypatch.setattr(evals_rl, "_rg_evals_governor", fake)
    monkeypatch.setattr(evals_rl, "_rg_evals_loader", None)

    limiter = evals_rl.UserRateLimiter()

    allowed, meta = await limiter.check_rate_limit(
        user_id="user-123",
        endpoint="/api/v1/evaluations",
        tokens_requested=123,
        estimated_cost=0.0,
    )

    assert allowed is True
    assert meta.get("policy_id") == "evals.free"
    assert meta.get("rate_limit_source") == "resource_governor"


@pytest.mark.asyncio
async def test_evaluations_rg_unavailable_fail_open(monkeypatch):
    """Phase 2: RG returns None → fail-open for eval/token caps, cost-only enforcement."""
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setattr(evals_rl, "_rg_evals_fallback_logged", False)
    monkeypatch.setattr(evals_rl, "_EVALS_DEPRECATION_WARNED", False)

    limiter = evals_rl.UserRateLimiter()

    async def _no_rg_decision(*args, **kwargs):  # noqa: ARG001
        return None

    monkeypatch.setattr(evals_rl, "_maybe_enforce_with_rg_evaluations", _no_rg_decision)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        allowed, meta = await limiter.check_rate_limit(
            user_id="user-123",
            endpoint="/api/v1/evaluations",
            tokens_requested=123,
            estimated_cost=0.0,
        )

        assert allowed is True
        assert meta.get("policy_id") == "evals.free"
        assert meta.get("rate_limit_source") == "resource_governor"

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1


@pytest.mark.asyncio
async def test_character_chat_rg_denies(monkeypatch):
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_CHARACTER_CHAT_ENFORCE_REQUESTS", "1")
    fake = _FakeGovernor(allowed=False, retry_after=3)
    monkeypatch.setattr(char_rl, "_rg_char_governor", fake)
    monkeypatch.setattr(char_rl, "_rg_char_loader", None)

    limiter = char_rl.CharacterRateLimiter(max_operations=100)

    with pytest.raises(Exception) as exc:
        await limiter.check_rate_limit(user_id=123, operation="character_op")

    assert "Rate limit exceeded" in str(exc.value)
    assert fake.reserved
    entity, categories, _op_id, tags = fake.reserved[-1]
    assert entity == "user:123"
    assert categories == {"requests": {"units": 1}}
    assert tags.get("module") == "character_chat"
    assert tags.get("operation") == "character_op"


@pytest.mark.asyncio
async def test_character_chat_invokes_rg_when_enabled(monkeypatch):
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_CHARACTER_CHAT_ENFORCE_REQUESTS", "1")
    fake = _FakeGovernor(allowed=True, retry_after=None)
    monkeypatch.setattr(char_rl, "_rg_char_governor", fake)
    monkeypatch.setattr(char_rl, "_rg_char_loader", None)

    limiter = char_rl.CharacterRateLimiter(max_operations=100, enabled=True)

    allowed, _remaining = await limiter.check_rate_limit(user_id=123, operation="character_op")

    assert allowed is True
    assert fake.reserved


@pytest.mark.asyncio
async def test_character_chat_rg_allows_bypasses_legacy_denies(monkeypatch):
    """When RG returns an allow decision, Character Chat should allow the request."""
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setenv("RG_CHARACTER_CHAT_ENFORCE_REQUESTS", "1")
    fake = _FakeGovernor(allowed=True, retry_after=None)
    monkeypatch.setattr(char_rl, "_rg_char_governor", fake)
    monkeypatch.setattr(char_rl, "_rg_char_loader", None)

    limiter = char_rl.CharacterRateLimiter(max_operations=1, window_seconds=3600)

    allowed, remaining = await limiter.check_rate_limit(user_id=123, operation="character_op")

    assert allowed is True
    assert isinstance(remaining, int)


@pytest.mark.asyncio
async def test_web_scraping_rg_denies(monkeypatch):
    monkeypatch.setenv("RG_ENABLED", "1")
    fake = _FakeGovernor(allowed=False, retry_after=1)
    monkeypatch.setattr(web_rl, "_rg_web_governor", fake)
    monkeypatch.setattr(web_rl, "_rg_web_loader", None)

    limiter = web_rl.RateLimiter(max_requests_per_second=100.0, max_requests_per_minute=1000, max_requests_per_hour=1000)

    # Ensure acquire completes quickly despite the artificial sleep; monkeypatch time if needed.
    start = asyncio.get_event_loop().time()
    await limiter.acquire()
    elapsed = asyncio.get_event_loop().time() - start

    assert fake.reserved
    entity, categories, _op_id, tags = fake.reserved[-1]
    assert entity == "service:web_scraping"
    assert categories == {"requests": {"units": 1}}
    assert tags.get("module") == "web_scraping"
    # We do not assert strict timing here to avoid flakiness, just that the call returned.


@pytest.mark.asyncio
async def test_web_scraping_rg_unavailable_fail_open(monkeypatch):
    """Phase 2: RG returns None → fail-open (no sleeps, no counters)."""
    monkeypatch.setenv("RG_ENABLED", "1")
    monkeypatch.setattr(web_rl, "_rg_web_fallback_logged", False)
    monkeypatch.setattr(web_rl, "_WEB_SCRAPING_DEPRECATION_WARNED", False)

    async def _no_rg_decision():
        return None

    monkeypatch.setattr(web_rl, "_maybe_enforce_with_rg_web_scraping", _no_rg_decision)

    limiter = web_rl.RateLimiter(
        max_requests_per_second=1000.0,
        max_requests_per_minute=1,
        max_requests_per_hour=1,
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        await limiter.acquire()

        # No assertion about _request_times — Phase 2 shim has no internal counters.
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1


@pytest.mark.asyncio
async def test_evaluations_rg_allows_still_enforces_cost_caps(monkeypatch):
    """
    When RG allows a request, the Evaluations limiter must still enforce
    cost caps locally (since RG does not handle cost limits).
    """
    monkeypatch.setenv("RG_ENABLED", "1")
    fake = _FakeGovernor(allowed=True, retry_after=None)
    monkeypatch.setattr(evals_rl, "_rg_evals_governor", fake)
    monkeypatch.setattr(evals_rl, "_rg_evals_loader", None)

    limiter = evals_rl.UserRateLimiter()

    usage_denied = False

    async def _deny_usage(*args, **kwargs):  # noqa: ARG001
        nonlocal usage_denied
        usage_denied = True
        return False, {
            "error": "Daily cost limit exceeded",
            "limit": 1.0,
            "used": 1.0,
            "requested": 0.5,
            "retry_after": 3600,
        }

    monkeypatch.setattr(limiter, "_reserve_request_usage", _deny_usage)

    allowed, meta = await limiter.check_rate_limit(
        user_id="user-cost-test",
        endpoint="/api/v1/evaluations",
        tokens_requested=0,
        estimated_cost=0.5,
    )

    assert usage_denied is True
    assert allowed is False
    assert "cost" in meta.get("error", "").lower()


@pytest.mark.asyncio
async def test_evaluations_rg_allows_with_zero_cost_skips_cost_check(monkeypatch):
    """
    Unified eval endpoints pass estimated_cost=0.0. When RG allows and
    cost is zero, the cost check should pass (no denial).
    """
    monkeypatch.setenv("RG_ENABLED", "1")
    fake = _FakeGovernor(allowed=True, retry_after=None)
    monkeypatch.setattr(evals_rl, "_rg_evals_governor", fake)
    monkeypatch.setattr(evals_rl, "_rg_evals_loader", None)

    limiter = evals_rl.UserRateLimiter()

    allowed, meta = await limiter.check_rate_limit(
        user_id="user-zero-cost",
        endpoint="/api/v1/evaluations",
        tokens_requested=100,
        estimated_cost=0.0,
    )

    assert allowed is True
    assert meta.get("rate_limit_source") == "resource_governor"
