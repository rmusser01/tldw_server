"""
Integration tests for /api/v1/chat/completions.

Behavior:
- If a real provider API key is detected in the environment, the test will
  make a real API call via the adapter to the provider.
- Otherwise, provider calls are mocked by monkeypatching the legacy handler.

This allows local/dev environments with keys to exercise a live integration
path while keeping CI deterministic and offline.
"""

import os
from typing import Any

import pytest

NETWORK_TESTS_ENABLED = os.getenv("ENABLE_NETWORK_TESTS", "").lower() in {"1", "true", "yes", "y", "on"}


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_obj: dict[str, Any] | None = None, lines: list[str] | None = None):
        self.status_code = status_code
        self._json = json_obj or {
            "id": "cmpl-test",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello there"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        self._lines = lines or [
            "data: {\"choices\":[{\"delta\":{\"content\":\"chunk\"}}]}\n\n",
            "data: [DONE]\n\n",
        ]

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            resp = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError("err", request=req, response=resp)

    def json(self):
        return self._json

    def iter_lines(self):
        yield from self._lines


class _FakeStreamCtx:
    def __init__(self, resp: _FakeResponse):
        self._resp = resp

    def __enter__(self):
        return self._resp

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, *args, **kwargs):
        return _FakeResponse()

    def stream(self, *args, **kwargs):
        return _FakeStreamCtx(_FakeResponse())


@pytest.fixture(autouse=True)
def _enable_adapters(monkeypatch):
    # Avoid endpoint-internal mock path that bypasses provider handlers
    monkeypatch.delenv("TEST_MODE", raising=False)
    # Ensure suite-level env overrides from other tests don't hijack base URL.
    # When a real API key is present, this test should talk to the real OpenAI API.
    for name in (
        "OPENAI_API_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "MOCK_OPENAI_BASE_URL",
        "CUSTOM_OPENAI_API_IP",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def _payload(stream: bool = False):
    return {
        "api_provider": "openai",
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": stream,
    }


def _real_key(provider: str) -> str | None:
    env_map = {
        "openai": ["OPENAI_API_KEY"],
    }
    for name in env_map.get(provider, []):
        val = os.getenv(name)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def test_chat_completions_non_streaming_via_adapter(monkeypatch, client, auth_token):
    real = _real_key("openai") if NETWORK_TESTS_ENABLED else None
    if real:
        # Use real key; do not monkeypatch provider call.
        r = client.post_with_auth("/api/v1/chat/completions", auth_token, json=_payload(stream=False))
        assert r.status_code == 200, f"Body: {r.text}"
        data = r.json()
        # Shape assertions only (provider-specific content may vary)
        assert data.get("object") == "chat.completion"
        assert isinstance(data.get("choices"), list) and len(data["choices"]) >= 1
    else:
        # Provide a test key and mock adapter HTTP client to avoid network
        monkeypatch.setenv("OPENAI_API_KEY", "sk-adapter-test-key")
        import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod
        import tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter as openrouter_mod
        monkeypatch.setattr(openai_mod, "http_client_factory", lambda *a, **k: _FakeClient())
        monkeypatch.setattr(openrouter_mod, "http_client_factory", lambda *a, **k: _FakeClient())
        r = client.post_with_auth("/api/v1/chat/completions", auth_token, json=_payload(stream=False))
        assert r.status_code == 200, f"Body: {r.text}"
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello there"


def test_chat_completions_streaming_via_adapter(monkeypatch, client, auth_token):
    real = _real_key("openai") if NETWORK_TESTS_ENABLED else None
    if real:
        from tldw_Server_API.tests._plugins.chat_fixtures import get_auth_headers
        headers = get_auth_headers(auth_token, getattr(client, "csrf_token", ""))
        with client.stream("POST", "/api/v1/chat/completions", json=_payload(stream=True), headers=headers) as resp:
            assert resp.status_code == 200
            ct = resp.headers.get("content-type", "").lower()
            assert ct.startswith("text/event-stream")
            lines = list(resp.iter_lines())
            # Should include chunks and single DONE
            assert any(line.startswith("data: ") and "[DONE]" not in line for line in lines)
            assert sum(1 for line in lines if line.strip().lower() == "data: [done]") == 1
    else:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-adapter-test-key")
        import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod
        import tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter as openrouter_mod
        monkeypatch.setattr(openai_mod, "http_client_factory", lambda *a, **k: _FakeClient())
        monkeypatch.setattr(openrouter_mod, "http_client_factory", lambda *a, **k: _FakeClient())

        from tldw_Server_API.tests._plugins.chat_fixtures import get_auth_headers
        headers = get_auth_headers(auth_token, getattr(client, "csrf_token", ""))
        with client.stream("POST", "/api/v1/chat/completions", json=_payload(stream=True), headers=headers) as resp:
            assert resp.status_code == 200
            ct = resp.headers.get("content-type", "").lower()
            assert ct.startswith("text/event-stream")
            lines = list(resp.iter_lines())
            # Should include chunks and single DONE
            assert any(line.startswith("data: ") and "[DONE]" not in line for line in lines)
            assert sum(1 for line in lines if line.strip().lower() == "data: [done]") == 1
