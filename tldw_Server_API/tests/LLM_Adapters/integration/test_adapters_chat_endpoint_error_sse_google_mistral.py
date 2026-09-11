"""
Endpoint pre-output error-path tests for Google (Gemini) and Mistral.

Provider failures raised before any stream output must be returned as a bounded
HTTP error before the response is handed off as SSE.
"""

from __future__ import annotations

from threading import Event

import pytest

# Ensure chat fixtures (client/auth) are registered as pytest fixtures
from tldw_Server_API.tests._plugins import chat_fixtures as _chat_pl  # noqa: F401


@pytest.fixture(autouse=True)
def _enable_adapters(monkeypatch):
    monkeypatch.setenv("STREAMS_UNIFIED", "1")
    # Disable TEST_MODE shunts
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("LOGURU_LEVEL", "ERROR")
    yield


def _payload(provider: str, *, stream: bool) -> dict:
    model = "gemini-2.5-pro" if provider == "google" else "mistral-large-latest"
    return {
        "api_provider": provider,
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": stream,
    }


@pytest.mark.integration
def test_chat_endpoint_streaming_error_google(monkeypatch, authenticated_client):
    """An eager Google stream failure is mapped before response handoff."""
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-gemini-test")
    adapter_called = Event()

    # Patch GoogleAdapter.stream to raise a ChatBadRequestError (normalized provider error)
    import tldw_Server_API.app.core.LLM_Calls.providers.google_adapter as google_mod
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError

    def _stream_raises(*args, **kwargs):
        adapter_called.set()
        raise ChatBadRequestError(provider="google", message="bad prompt")

    monkeypatch.setattr(google_mod.GoogleAdapter, "stream", _stream_raises, raising=True)

    response = authenticated_client.post(
        "/api/v1/chat/completions",
        json=_payload("google", stream=True),
    )

    assert response.status_code == 502
    assert response.json()["detail"]["error_code"] == "provider_unavailable"
    assert "bad prompt" not in response.text
    assert adapter_called.is_set()


@pytest.mark.integration
def test_chat_endpoint_streaming_error_mistral(monkeypatch, authenticated_client):
    """An eager Mistral stream failure is mapped before response handoff."""
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-mistral-test")
    adapter_called = Event()

    # Patch MistralAdapter.stream to raise a ChatProviderError (server-side)
    import tldw_Server_API.app.core.LLM_Calls.providers.mistral_adapter as mistral_mod
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError

    def _stream_raises(*args, **kwargs):
        adapter_called.set()
        raise ChatProviderError(provider="mistral", message="upstream 502", status_code=502)

    monkeypatch.setattr(mistral_mod.MistralAdapter, "stream", _stream_raises, raising=True)

    response = authenticated_client.post(
        "/api/v1/chat/completions",
        json=_payload("mistral", stream=True),
    )

    assert response.status_code == 502
    assert response.json()["detail"]["error_code"] == "provider_unavailable"
    assert "upstream 502" not in response.text
    assert adapter_called.is_set()
