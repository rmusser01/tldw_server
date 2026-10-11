"""/chat/completions records its own usage, so its provider call is marked (LLM usage accounting spec §3)."""

import pytest

pytestmark = pytest.mark.integration

from tldw_Server_API.app.api.v1.endpoints import chat as chat_ep
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context


def _post(client, auth_token, *, stream: bool):
    """Post a minimal local-llm chat request."""
    request_data = ChatCompletionRequest(
        model="local-llm",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        api_provider="local-llm",
        stream=stream,
    )
    settings = get_settings()
    headers = {"X-CSRF-Token": client.csrf_token}
    if settings.AUTH_MODE == "multi_user":
        headers["Authorization"] = auth_token
    else:
        headers["X-API-KEY"] = auth_token
    return client.post("/api/v1/chat/completions", json=request_data.model_dump(), headers=headers)


def test_chat_completions_marks_its_provider_call_as_self_recorded(
    client, auth_token, mock_chacha_db, setup_dependencies, configure_for_mock_server, monkeypatch
):
    """The non-streaming pipeline's provider call runs with records_itself=True."""
    seen: list[bool] = []

    def _dispatch(*args: object, **kwargs: object) -> dict:
        """Report the flag seen on the pool thread, then return a completion."""
        seen.append(current_llm_call_context().records_itself)
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "local-llm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(chat_ep, "perform_chat_api_call", _dispatch)
    monkeypatch.setitem(chat_ep.API_KEYS, "local-llm", "dummy-key")

    response = _post(client, auth_token, stream=False)

    assert response.status_code == 200, response.text
    assert seen == [True]


def test_chat_completions_streaming_marks_its_provider_call_as_self_recorded(
    client, auth_token, mock_chacha_db, setup_dependencies, configure_for_mock_server, monkeypatch
):
    """The streaming pipeline's provider call runs with records_itself=True."""
    seen: list[bool] = []

    def _dispatch(*args: object, **kwargs: object):
        """Report the flag seen on the pool thread, then return a canned stream."""
        seen.append(current_llm_call_context().records_itself)
        return iter(['data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', "data: [DONE]\n\n"])

    monkeypatch.setattr(chat_ep, "perform_chat_api_call", _dispatch)
    monkeypatch.setitem(chat_ep.API_KEYS, "local-llm", "dummy-key")

    response = _post(client, auth_token, stream=True)

    assert response.status_code == 200, response.text
    assert response.text
    assert seen == [True]
