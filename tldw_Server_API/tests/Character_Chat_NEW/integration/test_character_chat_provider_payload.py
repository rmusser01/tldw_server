"""Character completion coverage through real provider payload validation."""

from contextlib import contextmanager

import httpx
import pytest

from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import (
    CustomOpenAIAdapter,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("penalty", [None, 1.0, 1.1])
def test_complete_v2_neutral_penalty_reaches_custom_provider(test_client, auth_headers, monkeypatch, stream, penalty):
    """Default neutral sampling must not prevent a valid provider completion."""
    monkeypatch.setenv("CUSTOM_OPENAI_API_IP", "http://127.0.0.1:15973/v1")
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CUSTOM_OPENAI_API_MODEL", "local-uat-chat")

    def provider_response(**kwargs):
        return httpx.Response(
            200,
            request=httpx.Request(kwargs["method"], kwargs["url"]),
            json={"choices": [{"message": {"content": "Deterministic character reply"}}]},
        )

    @contextmanager
    def provider_stream(**kwargs):
        response = httpx.Response(
            200,
            request=httpx.Request(kwargs["method"], kwargs["url"]),
            content=('data: {"choices":[{"delta":{"content":"Deterministic character reply"}}]}\n\ndata: [DONE]\n\n'),
        )
        try:
            yield response
        finally:
            response.close()

    monkeypatch.setattr(CustomOpenAIAdapter, "http_fetcher", staticmethod(provider_response))
    monkeypatch.setattr(CustomOpenAIAdapter, "http_streamer", staticmethod(provider_stream))
    character = test_client.post(
        "/api/v1/characters/",
        headers=auth_headers,
        json={"name": "ProviderPayload", "first_message": "Ready."},
    )
    assert character.status_code == 201
    chat = test_client.post(
        "/api/v1/chats/",
        headers=auth_headers,
        json={"character_id": character.json()["id"]},
    )
    assert chat.status_code == 201
    body = {
        "provider": "custom-openai-api",
        "model": "custom-openai-api:local-uat-chat",
        "stream": stream,
        "save_to_db": True,
        "append_user_message": "Respond briefly.",
    }
    if penalty is not None:
        body["repetition_penalty"] = penalty
    response = test_client.post(
        f"/api/v1/chats/{chat.json()['id']}/complete-v2",
        headers=auth_headers,
        json=body,
    )
    if penalty == 1.1:
        # Unsupported non-neutral sampling must still fail validation.
        assert "Chat provider error" in response.text
        assert "Deterministic character reply" not in response.text
    else:
        assert response.status_code == 200
        assert "Deterministic character reply" in response.text
        assert "Chat provider error" not in response.text
