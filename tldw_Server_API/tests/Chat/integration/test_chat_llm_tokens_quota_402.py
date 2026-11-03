"""Chat 402 on the monthly LLM-token quota: Retry-After header, no dispatch (spec 2 review A7)."""

import pytest
from fastapi import HTTPException, status

pytestmark = pytest.mark.integration

from tldw_Server_API.app.api.v1.endpoints import chat as chat_ep
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
)
from tldw_Server_API.app.core import config_llm_providers
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.tests.helpers.provider_keys import api_keys_source


async def _tiny_llm_tokens_limit(_uid, key):
    """Only limits.llm_tokens_per_month is capped (at 1 token); every other key is unlimited."""
    return 1 if key == "limits.llm_tokens_per_month" else None


async def _month_already_spent(_uid):
    """The user's monthly usage is already far past any allowance."""
    return 1_000_000.0


def test_chat_completions_402_on_monthly_llm_token_quota(
    client, auth_token, mock_chacha_db, setup_dependencies, configure_for_mock_server, monkeypatch
):
    """At the monthly token allowance, /chat/completions returns 402 with Retry-After and never dispatches."""
    dispatch_calls: list[object] = []

    async def _recording_dispatch(*args: object, **kwargs: object) -> None:
        """Record a dispatch call; the quota check must short-circuit before this runs."""
        dispatch_calls.append((args, kwargs))
        raise AssertionError("LLM dispatch must not run once the monthly token quota is spent")

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.user_quota", _tiny_llm_tokens_limit)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.llm_tokens_this_month", _month_already_spent)
    monkeypatch.setattr(chat_ep, "perform_chat_api_call", _recording_dispatch)
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_orchestrator.chat_api_call", _recording_dispatch)
    monkeypatch.setattr(config_llm_providers, "get_api_keys", api_keys_source({"local-llm": "dummy-key"}))

    request_data = ChatCompletionRequest(
        model="local-llm",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello, how are you?")],
        api_provider="local-llm",
    )

    settings = get_settings()
    headers = {"X-CSRF-Token": client.csrf_token}
    if settings.AUTH_MODE == "multi_user":
        headers["Authorization"] = auth_token
    else:
        headers["X-API-KEY"] = auth_token

    response = client.post("/api/v1/chat/completions", json=request_data.model_dump(), headers=headers)

    assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED, response.text
    assert response.json()["detail"] == {
        "error": "limit_exceeded",
        "category": "llm_tokens_month",
        "current": 1_000_000,
        "limit": 1,
        "message": "Monthly LLM token limit reached",
    }
    assert int(response.headers["Retry-After"]) >= 1
    assert dispatch_calls == []


async def _daily_cost_limit(_uid, key):
    """Only limits.llm_cost_per_day_usd is set, at 50 cents."""
    return 0.5 if key == "limits.llm_cost_per_day_usd" else None


async def _spent_fifty_cents(_uid):
    """The user has spent 50 cents today."""
    return 0.5


@pytest.mark.parametrize("stream", [False, True])
def test_chat_completions_402_on_daily_llm_cost_quota(
    client, auth_token, mock_chacha_db, setup_dependencies, configure_for_mock_server, monkeypatch, stream
):
    """At the daily LLM cost cap, /chat/completions returns 402 before dispatch, streaming or not."""
    dispatch_calls: list[object] = []

    async def _recording_dispatch(*args: object, **kwargs: object) -> None:
        """Record a dispatch call; the quota check must short-circuit before this runs."""
        dispatch_calls.append((args, kwargs))
        raise AssertionError("LLM dispatch must not run once the daily cost quota is spent")

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.user_quota", _daily_cost_limit)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.llm_cost_today", _spent_fifty_cents)
    monkeypatch.setattr(chat_ep, "perform_chat_api_call", _recording_dispatch)
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_orchestrator.chat_api_call", _recording_dispatch)
    monkeypatch.setattr(config_llm_providers, "get_api_keys", api_keys_source({"local-llm": "dummy-key"}))

    request_data = ChatCompletionRequest(
        model="local-llm",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello, how are you?")],
        api_provider="local-llm",
        stream=stream,
    )
    settings = get_settings()
    headers = {"X-CSRF-Token": client.csrf_token}
    if settings.AUTH_MODE == "multi_user":
        headers["Authorization"] = auth_token
    else:
        headers["X-API-KEY"] = auth_token

    response = client.post("/api/v1/chat/completions", json=request_data.model_dump(), headers=headers)

    assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED, response.text
    assert response.json()["detail"]["category"] == "llm_cost_day"
    assert 1 <= int(response.headers["Retry-After"]) <= 86400
    assert dispatch_calls == []


def test_chat_completions_admits_an_over_limit_user_whose_provider_runs_on_their_own_key(
    client, auth_token, mock_chacha_db, setup_dependencies, configure_for_mock_server, monkeypatch
):
    """Past the monthly token allowance, a user with their own key for the provider still gets a reply (TASK-13955)."""
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    dispatched: list[str] = []

    async def _own_key(_uid, provider):
        """Stands in for the lookup answering True for the request's provider (the boundary is under test)."""
        return provider == "local-llm"

    def _dispatch(**kwargs):
        """Answer the call."""
        dispatched.append(str(kwargs.get("api_endpoint")))
        return {
            "id": "chatcmpl-own-key",
            "object": "chat.completion",
            "created": 0,
            "model": "local-llm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

    refuse_llm_work(monkeypatch)
    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    monkeypatch.setattr(chat_ep, "perform_chat_api_call", _dispatch)

    request_data = ChatCompletionRequest(
        model="local-llm",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello, how are you?")],
        api_provider="local-llm",
    )
    settings = get_settings()
    headers = {"X-CSRF-Token": client.csrf_token}
    if settings.AUTH_MODE == "multi_user":
        headers["Authorization"] = auth_token
    else:
        headers["X-API-KEY"] = auth_token

    response = client.post("/api/v1/chat/completions", json=request_data.model_dump(), headers=headers)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["choices"][0]["message"]["content"] == "hi"
    assert dispatched == ["local-llm"]


def test_chat_completions_answers_402_when_the_call_itself_is_refused(
    client, auth_token, mock_chacha_db, setup_dependencies, configure_for_mock_server, monkeypatch
):
    """A refusal raised at the call, as after an unmarked own-key admission, reaches the client as 402, not 500 (TASK-13955)."""
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    def _refused_at_the_call(**_kwargs):
        """What the adapters' check at the call raises for a call not on the user's own key."""
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    monkeypatch.setattr(chat_ep, "perform_chat_api_call", _refused_at_the_call)

    request_data = ChatCompletionRequest(
        model="local-llm",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello, how are you?")],
        api_provider="local-llm",
    )
    settings = get_settings()
    headers = {"X-CSRF-Token": client.csrf_token}
    if settings.AUTH_MODE == "multi_user":
        headers["Authorization"] = auth_token
    else:
        headers["X-API-KEY"] = auth_token

    response = client.post("/api/v1/chat/completions", json=request_data.model_dump(), headers=headers)

    assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED, response.text
    assert response.json()["detail"]["category"] == "llm_tokens_month"


class _PlanSpent:
    """The hosted billing plan's LLM token check, with the org's plan already spent."""

    def __init__(self, *_args, **_kwargs):
        """Any org, category and estimate."""

    async def __aenter__(self):
        """Refuse, as LimitEnforcer does past the plan's llm_tokens_month."""
        raise HTTPException(status_code=402, detail={"error": "limit_exceeded", "category": "llm_tokens_month"})


def _on_a_spent_hosted_plan(client, monkeypatch) -> None:
    """Billing checks on for org 7, whose plan refuses every LLM request (the hosted product)."""
    monkeypatch.setattr(chat_ep, "enforcement_enabled", lambda: True)
    monkeypatch.setattr(chat_ep, "LimitEnforcer", _PlanSpent)
    monkeypatch.setitem(client.app.dependency_overrides, chat_ep.get_billing_org_id, lambda: 7)


@pytest.mark.parametrize("own_key", [True, False], ids=["own-key", "server-key"])
def test_the_hosted_plan_does_not_check_a_request_on_the_users_own_key(
    client, auth_token, mock_chacha_db, setup_dependencies, configure_for_mock_server, monkeypatch, own_key
):
    """A spent org plan refuses a server-key request but not one whose provider runs on the user's own key (TASK-13955)."""
    dispatched: list[str] = []

    async def _own_key(_uid, provider):
        """Stands in for the lookup: the request's provider runs on the user's own key, or not."""
        return own_key and provider == "local-llm"

    def _dispatch(**kwargs):
        """Answer the call."""
        dispatched.append(str(kwargs.get("api_endpoint")))
        return {
            "id": "chatcmpl-plan",
            "object": "chat.completion",
            "created": 0,
            "model": "local-llm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

    _on_a_spent_hosted_plan(client, monkeypatch)
    monkeypatch.setattr(chat_ep, "users_own_key_serves", _own_key)
    monkeypatch.setattr(chat_ep, "perform_chat_api_call", _dispatch)

    request_data = ChatCompletionRequest(
        model="local-llm",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello, how are you?")],
        api_provider="local-llm",
    )
    settings = get_settings()
    headers = {"X-CSRF-Token": client.csrf_token}
    if settings.AUTH_MODE == "multi_user":
        headers["Authorization"] = auth_token
    else:
        headers["X-API-KEY"] = auth_token

    response = client.post("/api/v1/chat/completions", json=request_data.model_dump(), headers=headers)

    if own_key:
        assert response.status_code == status.HTTP_200_OK, response.text
        assert dispatched == ["local-llm"]
    else:
        assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED, response.text
        assert response.json()["detail"]["category"] == "llm_tokens_month"
        assert dispatched == []


def test_a_hosted_over_limit_request_on_the_users_own_key_looks_the_key_up_once(
    client, auth_token, mock_chacha_db, setup_dependencies, configure_for_mock_server, monkeypatch
):
    """The hosted plan's skip and the per-user limits' admission share one own-key lookup (TASK-13955 review)."""
    from tldw_Server_API.app.api.v1.API_Deps import usage_quota_deps
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work

    asked: list[str] = []

    async def _own_key(_uid, provider):
        """The request's provider runs on the user's own key; each lookup is noted."""
        asked.append(provider)
        return provider == "local-llm"

    def _dispatch(**_kwargs):
        """Answer the call."""
        return {
            "id": "chatcmpl-once",
            "object": "chat.completion",
            "created": 0,
            "model": "local-llm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

    _on_a_spent_hosted_plan(client, monkeypatch)
    refuse_llm_work(monkeypatch)
    monkeypatch.setattr(chat_ep, "users_own_key_serves", _own_key)
    monkeypatch.setattr(usage_quota_deps, "users_own_key_serves", _own_key)
    monkeypatch.setattr(chat_ep, "perform_chat_api_call", _dispatch)

    request_data = ChatCompletionRequest(
        model="local-llm",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello, how are you?")],
        api_provider="local-llm",
    )
    settings = get_settings()
    headers = {"X-CSRF-Token": client.csrf_token}
    if settings.AUTH_MODE == "multi_user":
        headers["Authorization"] = auth_token
    else:
        headers["X-API-KEY"] = auth_token

    response = client.post("/api/v1/chat/completions", json=request_data.model_dump(), headers=headers)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert asked == ["local-llm"]
