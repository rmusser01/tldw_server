"""
Endpoint pre-output error tests for OpenAI, Anthropic, Groq, and OpenRouter.

Provider failures raised before any stream output must be returned as bounded
HTTP errors before the response is handed off as SSE.
"""

from __future__ import annotations

from threading import Event

import pytest

# Ensure chat fixtures (client/auth) are registered as pytest fixtures
from tldw_Server_API.tests._plugins import chat_fixtures as _chat_pl  # noqa: F401


@pytest.fixture(autouse=True)
def _enable_adapters(monkeypatch):
    monkeypatch.setenv("STREAMS_UNIFIED", "1")
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("LOGURU_LEVEL", "ERROR")
    yield


def _payload(provider: str) -> dict:
    model = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet",
        "groq": "llama3-groq-8b",
        "openrouter": "openrouter/auto",
    }[provider]
    return {
        "api_provider": provider,
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }


@pytest.mark.integration
def test_chat_endpoint_streaming_error_openai(monkeypatch, authenticated_client):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    adapter_called = Event()

    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError

    def _stream_raises(*args, **kwargs):
        adapter_called.set()
        raise ChatBadRequestError(provider="openai", message="invalid input")

    monkeypatch.setattr(openai_mod.OpenAIAdapter, "stream", _stream_raises, raising=True)

    response = authenticated_client.post(
        "/api/v1/chat/completions",
        json=_payload("openai"),
    )

    assert response.status_code == 502
    assert response.json()["detail"]["error_code"] == "provider_unavailable"
    assert "invalid input" not in response.text
    assert adapter_called.is_set()


@pytest.mark.integration
def test_chat_endpoint_streaming_error_anthropic(monkeypatch, authenticated_client):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    adapter_called = Event()

    import tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter as ant_mod
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError

    def _stream_raises(*args, **kwargs):
        adapter_called.set()
        raise ChatProviderError(provider="anthropic", message="server error", status_code=500)

    monkeypatch.setattr(ant_mod.AnthropicAdapter, "stream", _stream_raises, raising=True)

    response = authenticated_client.post(
        "/api/v1/chat/completions",
        json=_payload("anthropic"),
    )

    assert response.status_code == 502
    assert response.json()["detail"]["error_code"] == "provider_unavailable"
    assert "server error" not in response.text
    assert adapter_called.is_set()


@pytest.mark.integration
def test_chat_endpoint_streaming_error_groq(monkeypatch, authenticated_client):
    monkeypatch.setenv("GROQ_API_KEY", "sk-groq-test")
    adapter_called = Event()

    import tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter as groq_mod
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatRateLimitError

    def _stream_raises(*args, **kwargs):
        adapter_called.set()
        raise ChatRateLimitError(provider="groq", message="too many requests")

    monkeypatch.setattr(groq_mod.GroqAdapter, "stream", _stream_raises, raising=True)

    response = authenticated_client.post(
        "/api/v1/chat/completions",
        json=_payload("groq"),
    )

    assert response.status_code == 502
    assert response.json()["detail"]["error_code"] == "provider_unavailable"
    assert "too many requests" not in response.text
    assert adapter_called.is_set()


@pytest.mark.integration
def test_chat_endpoint_streaming_error_openrouter(monkeypatch, authenticated_client):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    adapter_called = Event()

    import tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter as or_mod
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError

    def _stream_raises(*args, **kwargs):
        adapter_called.set()
        raise ChatAuthenticationError(provider="openrouter", message="bad key")

    monkeypatch.setattr(or_mod.OpenRouterAdapter, "stream", _stream_raises, raising=True)

    response = authenticated_client.post(
        "/api/v1/chat/completions",
        json=_payload("openrouter"),
    )

    assert response.status_code == 502
    assert response.json()["detail"]["error_code"] == "provider_authentication_failed"
    assert "bad key" not in response.text
    assert adapter_called.is_set()
