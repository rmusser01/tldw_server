import json
import os
from typing import Any, Dict

import httpx
import pytest

from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_google_gemini_tools_and_inline_image_mapping(monkeypatch):
    # Force native path and enable tools + inline image mapping
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_GOOGLE", "1")
    monkeypatch.setenv("LLM_ADAPTERS_GEMINI_TOOLS_BETA", "1")
    # PYTEST_CURRENT_TEST is present in pytest, but set explicitly for safety
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_google_gemini_tools_and_inline_image_mapping")

    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured
        assert request.method == "POST"
        # generateContent path
        assert request.url.path.endswith(":generateContent")
        payload = json.loads(request.content.decode("utf-8"))
        captured = payload
        # Validate tools mapping
        tools = payload.get("tools") or []
        assert tools and isinstance(tools, list)
        fdecls = tools[0].get("functionDeclarations")
        assert fdecls and isinstance(fdecls, list)
        assert fdecls[0]["name"] == "do_something"
        # Validate inline image mapping
        contents = payload.get("contents") or []
        assert contents and isinstance(contents, list)
        parts = contents[0].get("parts") or []
        # Should include both text and inlineData
        assert any("text" in p for p in parts)
        assert any("inlineData" in p for p in parts)
        # Return minimal Gemini response
        data = {
            "responseId": "resp_123",
            "candidates": [
                {"content": {"parts": [{"text": "Hello from Gemini"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
        }
        return httpx.Response(200, json=data)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    # Patch both the module alias and the http_client factory
    import tldw_Server_API.app.core.http_client as hc
    monkeypatch.setattr(hc, "create_client", fake_create_client)
    import tldw_Server_API.app.core.LLM_Calls.providers.google_adapter as gmod
    monkeypatch.setattr(gmod, "_hc_create_client", fake_create_client)
    monkeypatch.setattr(gmod, "http_client_factory", fake_create_client)

    adapter = GoogleAdapter()
    # Messages with text and data: URL image part
    request = {
        "model": "gemini-2.5-pro",
        "api_key": "sk-test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,aGVsbG8="
                        },
                    },
                ],
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "do_something",
                    "description": "test function",
                    "parameters": {
                        "type": "object",
                        "properties": {"foo": {"type": "string"}},
                        "required": ["foo"],
                    },
                },
            }
        ],
    }

    out = adapter.chat(request)
    # OpenAI-shaped
    assert out["choices"][0]["message"]["content"] == "Hello from Gemini"


def test_google_error_normalization_auth(monkeypatch):
    # Force native path to hit error normalization
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_GOOGLE", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_google_error_normalization_auth")

    def handler(request: httpx.Request) -> httpx.Response:
        data = {"error": {"status": "UNAUTHENTICATED", "code": 401, "message": "invalid api key"}}
        return httpx.Response(401, json=data)

    def fake_create_client(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=_mock_transport(handler))

    import tldw_Server_API.app.core.http_client as hc
    monkeypatch.setattr(hc, "create_client", fake_create_client)
    import tldw_Server_API.app.core.LLM_Calls.providers.google_adapter as gmod
    monkeypatch.setattr(gmod, "_hc_create_client", fake_create_client)
    monkeypatch.setattr(gmod, "http_client_factory", fake_create_client)

    adapter = GoogleAdapter()
    with pytest.raises(Exception) as ei:
        adapter.chat({"model": "gemini-2.5-pro", "api_key": "bad", "messages": [{"role": "user", "content": "hi"}]})
    assert str(ei.value) == "Authentication failed with the chat provider."
