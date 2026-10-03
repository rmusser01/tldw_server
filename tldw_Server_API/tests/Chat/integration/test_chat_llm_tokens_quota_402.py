"""Chat 402 on the monthly LLM-token quota: Retry-After header, no dispatch (spec 2 review A7)."""

from fastapi import status

from tldw_Server_API.app.api.v1.endpoints import chat as chat_ep
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings


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
    monkeypatch.setattr(chat_ep, "llm_tokens_this_month", _month_already_spent)
    monkeypatch.setattr(chat_ep, "perform_chat_api_call", _recording_dispatch)
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_orchestrator.chat_api_call", _recording_dispatch)
    monkeypatch.setitem(chat_ep.API_KEYS, "local-llm", "dummy-key")

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
    assert int(response.headers["Retry-After"]) >= 1
    assert dispatch_calls == []
