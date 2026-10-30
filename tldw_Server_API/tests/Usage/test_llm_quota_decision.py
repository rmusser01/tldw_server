"""The LLM quota decision, the 402 error and the boundary helpers (LLM usage accounting spec §4, §4a)."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage import quota_checks
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context

pytestmark = pytest.mark.unit


def _set_limits(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokens: float | None = None,
    cost_day: float | None = None,
    cost_month: float | None = None,
    used_tokens: float = 0.0,
    spent_today: float = 0.0,
    spent_month: float = 0.0,
) -> list[str]:
    """Give the user these limits and this usage; return the list of counters that ran."""
    limits = {
        "limits.llm_tokens_per_month": tokens,
        "limits.llm_cost_per_day_usd": cost_day,
        "limits.llm_cost_per_month_usd": cost_month,
    }
    ran: list[str] = []

    async def _user_quota(_uid: object, key: str) -> float | None:
        """The configured limit for ``key``."""
        return limits.get(key)

    def _counter(name: str, value: float):
        """A counter that records that it ran."""

        async def _read(_uid: object) -> float:
            ran.append(name)
            return value

        return _read

    monkeypatch.setattr(quota_checks, "user_quota", _user_quota)
    monkeypatch.setattr(quota_checks, "llm_tokens_this_month", _counter("tokens", used_tokens))
    monkeypatch.setattr(quota_checks, "llm_cost_today", _counter("cost_today", spent_today))
    monkeypatch.setattr(quota_checks, "llm_cost_this_month", _counter("cost_month", spent_month))
    return ran


async def test_no_limits_admit_without_reading_any_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset limits cost no query."""
    ran = _set_limits(monkeypatch)
    assert await quota_checks.llm_quota_refusal(7, 500) is None
    assert ran == []


async def test_tokens_refuse_when_this_request_would_pass_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Used plus requested above the limit refuses; at the limit admits."""
    _set_limits(monkeypatch, tokens=100, used_tokens=99)
    assert await quota_checks.llm_quota_refusal(7, 1) is None
    refusal = await quota_checks.llm_quota_refusal(7, 2)
    assert refusal is not None
    assert (refusal.category, refusal.current, refusal.limit, refusal.message) == (
        "llm_tokens_month", 99, 100, "Monthly LLM token limit reached",
    )
    assert 1 <= refusal.retry_after <= 31 * 86400


@pytest.mark.parametrize(
    ("limits", "category", "message", "max_retry"),
    [
        ({"cost_day": 0.5, "spent_today": 0.5}, "llm_cost_day", "Daily LLM cost limit reached", 86400),
        ({"cost_month": 2.0, "spent_month": 2.5}, "llm_cost_month", "Monthly LLM cost limit reached", 31 * 86400),
    ],
)
async def test_cost_refuses_once_spend_reaches_the_cap(
    monkeypatch: pytest.MonkeyPatch, limits: dict, category: str, message: str, max_retry: int
) -> None:
    """Spend at or over a cost cap refuses with that cap's category, message and reset time."""
    _set_limits(monkeypatch, **limits)
    refusal = await quota_checks.llm_quota_refusal(7)
    assert refusal is not None
    assert (refusal.category, refusal.message) == (category, message)
    assert refusal.current == max(limits.get("spent_today", 0), limits.get("spent_month", 0))
    assert 1 <= refusal.retry_after <= max_retry


async def test_spend_under_the_cap_admits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cent under the daily cap is admitted."""
    _set_limits(monkeypatch, cost_day=0.5, spent_today=0.49)
    assert await quota_checks.llm_quota_refusal(7) is None


@pytest.mark.parametrize("limits", [{"tokens": 0}, {"cost_day": 0}, {"cost_month": 0}])
async def test_zero_blocks(monkeypatch: pytest.MonkeyPatch, limits: dict) -> None:
    """A limit of 0 refuses a user with no usage at all."""
    _set_limits(monkeypatch, **limits)
    assert await quota_checks.llm_quota_refusal(7) is not None


async def test_tokens_are_checked_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Over both the token and the daily cost limit, the token refusal is reported and no cost counter runs."""
    ran = _set_limits(monkeypatch, tokens=10, used_tokens=50, cost_day=1.0, spent_today=5.0)
    refusal = await quota_checks.llm_quota_refusal(7)
    assert refusal is not None and refusal.category == "llm_tokens_month"
    assert ran == ["tokens"]


async def test_a_counter_failure_admits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cost counter that fails lets the request through, as every quota counter does."""
    _set_limits(monkeypatch, cost_day=0.5)

    async def _broken(_uid: object) -> float:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(quota_checks, "llm_cost_today", _broken)
    assert await quota_checks.llm_quota_refusal(7) is None


def test_the_error_is_the_public_402() -> None:
    """The error carries the body /chat/completions returns today, and Retry-After."""
    error = LLMQuotaExceededError(
        category="llm_cost_day", current=0.5, limit=0.5, message="Daily LLM cost limit reached", retry_after=60
    )
    assert error.status_code == 402
    assert error.detail == {
        "error": "limit_exceeded",
        "category": "llm_cost_day",
        "current": 0.5,
        "limit": 0.5,
        "message": "Daily LLM cost limit reached",
    }
    assert error.headers == {"Retry-After": "60"}


async def test_enforce_marks_the_task_only_when_it_admits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A passing check marks the rest of the task prechecked; a refusal raises and marks nothing."""

    async def _admitted() -> bool:
        await usage_quota_deps.enforce_llm_quota(7)
        return current_llm_call_context().prechecked

    async def _refused() -> bool:
        with pytest.raises(LLMQuotaExceededError) as caught:
            await usage_quota_deps.enforce_llm_quota(7)
        assert caught.value.detail["category"] == "llm_tokens_month"
        return current_llm_call_context().prechecked

    _set_limits(monkeypatch)
    assert await asyncio.create_task(_admitted()) is True
    _set_limits(monkeypatch, tokens=0)
    assert await asyncio.create_task(_refused()) is False
    assert current_llm_call_context().prechecked is False


def _app_with_a_quota_dependency() -> FastAPI:
    """A one-route app whose route depends on require_llm_quota and streams what its body sees."""
    app = FastAPI()
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=7)

    @app.get("/stream", dependencies=[Depends(usage_quota_deps.require_llm_quota)])
    async def _stream() -> StreamingResponse:
        async def _body():
            yield f"prechecked={current_llm_call_context().prechecked}"

        return StreamingResponse(_body(), media_type="text/plain")

    return app


def test_the_dependency_mark_reaches_a_streaming_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mark set by require_llm_quota is still in effect when the streaming response body runs."""
    _set_limits(monkeypatch)
    with TestClient(_app_with_a_quota_dependency()) as client:
        response = client.get("/stream")
    assert (response.status_code, response.text) == (200, "prechecked=True")


def test_the_dependency_refuses_with_402_before_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """An over-quota user gets the 402 body and Retry-After, and the route never runs."""
    _set_limits(monkeypatch, cost_month=1.0, spent_month=3.0)
    with TestClient(_app_with_a_quota_dependency()) as client:
        response = client.get("/stream")
    assert response.status_code == 402
    assert response.json()["detail"]["category"] == "llm_cost_month"
    assert int(response.headers["Retry-After"]) >= 1


async def test_an_over_limit_user_whose_provider_runs_on_their_own_key_is_let_through_unmarked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With their own key for the provider, the boundary admits without the mark; without it, it refuses (TASK-13955)."""
    asked: list[tuple[object, str]] = []

    async def _own_key(user_id: object, provider: str) -> bool:
        """The user has their own key for openai only."""
        asked.append((user_id, provider))
        return provider == "openai"

    async def _enforce(provider: str | None) -> bool:
        await usage_quota_deps.enforce_llm_quota(7, provider=provider)
        return current_llm_call_context().prechecked

    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    _set_limits(monkeypatch, tokens=0)
    assert await asyncio.create_task(_enforce("openai")) is False
    for provider in ("anthropic", None):
        with pytest.raises(LLMQuotaExceededError):
            await usage_quota_deps.enforce_llm_quota(7, provider=provider)
    _set_limits(monkeypatch)
    assert await asyncio.create_task(_enforce("openai")) is True
    assert asked == [(7, "openai"), (7, "anthropic")]


@pytest.mark.parametrize(
    ("requested", "byok_on", "allowlisted", "row", "base_url", "expected"),
    [
        ("OpenAI", True, True, {"id": 1}, None, True),
        ("OpenAI", True, True, None, None, False),
        ("OpenAI", False, True, {"id": 1}, None, False),
        ("OpenAI", True, False, {"id": 1}, None, False),
        ("OpenAI", True, True, RuntimeError("store down"), None, False),
        ("custom-openai-api", True, True, {"id": 1}, None, False),
        ("ollama", True, True, {"id": 1}, None, False),
        ("OpenAI", True, True, {"id": 1}, "https://api.openai.com/v1", True),
        ("OpenAI", True, True, {"id": 1}, "http://operator-gpu.internal:8000/v1", False),
    ],
    ids=[
        "own-key", "no-key", "byok-off", "not-allowlisted", "lookup-fails", "operator-endpoint", "local-server",
        "vendor-base-url", "operator-base-url",
    ],
)
async def test_users_own_key_serves_mirrors_the_first_test_of_credential_resolution(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    byok_on: bool,
    allowlisted: bool,
    row: object,
    base_url: str | None,
    expected: bool,
) -> None:
    """A billed provider on its vendor's host, BYOK on, the provider allowlisted and an active key; else False."""
    from tldw_Server_API.app.core.AuthNZ import byok_helpers
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo

    async def _pool() -> object:
        return object()

    async def _fetch(_self: object, user_id: int, provider: str, **_kwargs: object) -> object:
        assert (user_id, provider) == (7, requested.lower())
        if isinstance(row, Exception):
            raise row
        return row

    snapshot = {"openai_api": {"api_base_url": base_url}} if base_url else {}
    monkeypatch.setattr(byok_helpers, "is_byok_enabled", lambda: byok_on)
    monkeypatch.setattr(byok_helpers, "is_provider_allowlisted", lambda _provider: allowlisted)
    monkeypatch.setattr(byok_helpers, "load_server_config_snapshot", lambda: snapshot)
    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.get_db_pool", _pool)
    monkeypatch.setattr(AuthnzUserProviderSecretsRepo, "fetch_secret_for_active_user", _fetch)
    assert await usage_quota_deps.users_own_key_serves("7", requested) is expected
