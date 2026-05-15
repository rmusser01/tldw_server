from __future__ import annotations

import json
from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.unit


class _FakeResponse:
    def __init__(self, json_obj: dict[str, Any] | None = None):
        self.status_code = 200
        self._json = json_obj or {"object": "chat.completion", "choices": [{"message": {"content": "ok"}}]}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._json


class _CapturingClient:
    def __init__(self, captured: dict[str, Any]):
        self._captured = captured

    def __enter__(self) -> "_CapturingClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
        self._captured["url"] = url
        self._captured["json"] = json
        self._captured["headers"] = headers
        return _FakeResponse()


def _capture_factory(captured: dict[str, Any]):
    return lambda *args, **kwargs: _CapturingClient(captured)


def test_openai_cache_intent_adds_only_documented_cache_fields(monkeypatch) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod

    captured: dict[str, Any] = {}
    monkeypatch.setattr(openai_mod, "http_client_factory", _capture_factory(captured), raising=True)

    response = OpenAIAdapter().chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-5.4",
            "api_key": "sk-test",
            "billing_prompt_cache_intent": {
                "enabled": True,
                "scope": ["system"],
                "ttl_seconds": 86400,
                "static_segment_fingerprint": "prompt-v1:abc123",
            },
        }
    )

    payload = captured["json"]
    assert payload["prompt_cache_key"] == "tldw:prompt-v1:abc123"
    assert payload["prompt_cache_retention"] == "24h"
    assert "billing_prompt_cache_intent" not in payload
    assert response["tldw_cache_intent"]["cache_intent_requested"] is True
    assert response["tldw_cache_intent"]["cache_intent_applied"] is True
    assert response["tldw_cache_intent"]["provider_usage_authoritative"] is False


def test_openai_without_cache_intent_preserves_payload_shape(monkeypatch) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod

    captured: dict[str, Any] = {}
    monkeypatch.setattr(openai_mod, "http_client_factory", _capture_factory(captured), raising=True)

    response = OpenAIAdapter().chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4o-mini",
            "api_key": "sk-test",
        }
    )

    payload = captured["json"]
    assert "prompt_cache_key" not in payload
    assert "prompt_cache_retention" not in payload
    assert "tldw_cache_intent" not in response


def test_anthropic_cache_intent_marks_system_content_block(monkeypatch) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter import AnthropicAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter as anthropic_mod

    captured: dict[str, Any] = {}
    monkeypatch.setattr(anthropic_mod, "http_client_factory", _capture_factory(captured), raising=True)

    response = AnthropicAdapter().chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "system_message": "Stable character card",
            "model": "claude-sonnet-4.6",
            "api_key": "sk-test",
            "max_tokens": 64,
            "billing_prompt_cache_intent": {
                "enabled": True,
                "scope": ["system"],
                "ttl_seconds": 3600,
                "static_segment_fingerprint": "prompt-v1:anthropic",
            },
        }
    )

    payload = captured["json"]
    assert payload["system"] == [
        {
            "type": "text",
            "text": "Stable character card",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    assert "billing_prompt_cache_intent" not in payload
    assert response["tldw_cache_intent"]["applied_fields"] == ["system.cache_control"]


def test_google_cache_intent_requires_explicit_cached_content_reference(monkeypatch) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.google_adapter as google_mod

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "responseId": "resp_1",
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1},
            },
        )

    monkeypatch.setattr(
        google_mod,
        "http_client_factory",
        lambda *args, **kwargs: httpx.Client(transport=httpx.MockTransport(handler)),
        raising=True,
    )

    response = GoogleAdapter().chat(
        {
            "messages": [{"role": "user", "content": "Summarize"}],
            "model": "gemini-3-flash-preview",
            "api_key": "sk-test",
            "billing_prompt_cache_intent": {
                "enabled": True,
                "scope": ["system"],
                "provider_hint": {"cached_content": "cachedContents/cache-123"},
            },
        }
    )

    assert captured["json"]["cachedContent"] == "cachedContents/cache-123"
    assert response["tldw_cache_intent"]["cache_intent_applied"] is True


def test_google_cache_intent_without_cached_content_is_diagnostic_noop(monkeypatch) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.google_adapter as google_mod

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "responseId": "resp_1",
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1},
            },
        )

    monkeypatch.setattr(
        google_mod,
        "http_client_factory",
        lambda *args, **kwargs: httpx.Client(transport=httpx.MockTransport(handler)),
        raising=True,
    )

    response = GoogleAdapter().chat(
        {
            "messages": [{"role": "user", "content": "Summarize"}],
            "model": "gemini-3-flash-preview",
            "api_key": "sk-test",
            "billing_prompt_cache_intent": {"enabled": True, "scope": ["system"]},
        }
    )

    assert "cachedContent" not in captured["json"]
    assert response["tldw_cache_intent"]["cache_intent_requested"] is True
    assert response["tldw_cache_intent"]["cache_intent_applied"] is False
    assert response["tldw_cache_intent"]["reason"] == "gemini_cached_content_reference_required"


def test_openrouter_cache_intent_whitelists_provider_routing_metadata(monkeypatch) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter import OpenRouterAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter as openrouter_mod

    captured: dict[str, Any] = {}
    monkeypatch.setattr(openrouter_mod, "http_client_factory", _capture_factory(captured), raising=True)

    response = OpenRouterAdapter().chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "system_message": "Stable system",
            "model": "anthropic/claude-sonnet-4.6",
            "api_key": "sk-test",
            "billing_prompt_cache_intent": {
                "enabled": True,
                "scope": ["system"],
                "ttl_seconds": 3600,
                "provider_hint": {
                    "openrouter": {
                        "cache_control": "automatic",
                        "provider": {
                            "order": ["Anthropic"],
                            "allow_fallbacks": False,
                            "api_key": "must-not-forward",
                        },
                    }
                },
            },
        }
    )

    payload = captured["json"]
    assert payload["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert payload["provider"] == {"order": ["Anthropic"], "allow_fallbacks": False}
    assert "api_key" not in payload["provider"]
    assert "billing_prompt_cache_intent" not in payload
    assert response["tldw_cache_intent"]["applied_fields"] == ["cache_control", "provider"]


def test_openrouter_extra_body_cache_control_is_not_reported_as_intent(monkeypatch) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter import OpenRouterAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter as openrouter_mod

    captured: dict[str, Any] = {}
    monkeypatch.setattr(openrouter_mod, "http_client_factory", _capture_factory(captured), raising=True)

    response = OpenRouterAdapter().chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "anthropic/claude-sonnet-4.6",
            "api_key": "sk-test",
            "extra_body": {"cache_control": {"type": "ephemeral"}},
        }
    )

    assert captured["json"]["cache_control"] == {"type": "ephemeral"}
    assert "tldw_cache_intent" not in response
