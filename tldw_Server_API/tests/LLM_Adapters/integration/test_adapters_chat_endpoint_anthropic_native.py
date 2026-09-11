"""
Integration tests for /api/v1/chat/completions using Anthropic adapter native HTTP.

Behavior:
- If ANTHROPIC_API_KEY is set, make a real API call via the adapter.
- Otherwise, replace the adapter HTTP client to avoid network and keep deterministic behavior.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

NETWORK_TESTS_ENABLED = os.getenv("ENABLE_NETWORK_TESTS", "").lower() in {"1", "true", "yes", "y", "on"}

_AUTH_HEADERS = {"Authorization": f"Bearer {os.environ.get('SINGLE_USER_API_KEY', 'test-api-key-12345')}"}

class _FakeResponse:
    def __init__(self, status_code: int = 200, json_obj: dict[str, Any] | None = None, lines: list[str] | None = None):
        self.status_code = status_code
        self._json = json_obj or {"type": "message", "content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 1}}
        self._lines = lines or [
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"a\"}}",
            "data: [DONE]",
        ]

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            resp = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError("err", request=req, response=resp)

    def json(self):
        return self._json

    def iter_lines(self):
        yield from self._lines


class _FakeStreamCtx:
    def __init__(self, r: _FakeResponse):
        self._r = r

    def __enter__(self):
        return self._r

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
        return _FakeResponse(200)

    def stream(self, method: str, url: str, json: dict[str, Any], headers: dict[str, str]):
        return _FakeStreamCtx(_FakeResponse(200))


@pytest.fixture(autouse=True)
def _enable_native(monkeypatch):
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_ANTHROPIC", "1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
    # Avoid TEST_MODE mock path so endpoint calls provider
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("LOGURU_LEVEL", "ERROR")
    yield


def _payload(stream: bool = False):
    return {
        "api_provider": "anthropic",
        "model": "claude-sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": stream,
    }


def test_chat_completions_anthropic_native_non_streaming(monkeypatch, client_user_only):
    real = os.getenv("ANTHROPIC_API_KEY") if NETWORK_TESTS_ENABLED else None
    if real:
        client = client_user_only
        r = client.post("/api/v1/chat/completions", json=_payload(stream=False), headers=_AUTH_HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data.get("object") == "chat.completion"
        assert isinstance(data.get("choices"), list)
    else:
        # Mock httpx client to avoid network when no key
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        import tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter as anthropic_mod
        monkeypatch.setattr(anthropic_mod, "http_client_factory", lambda *a, **k: _FakeClient())

        client = client_user_only
        r = client.post("/api/v1/chat/completions", json=_payload(stream=False), headers=_AUTH_HEADERS)
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "chat.completion"


def test_chat_completions_anthropic_native_streaming(monkeypatch, client_user_only):
    real = os.getenv("ANTHROPIC_API_KEY") if NETWORK_TESTS_ENABLED else None
    if real:
        client = client_user_only
        with client.stream("POST", "/api/v1/chat/completions", json=_payload(stream=True), headers=_AUTH_HEADERS) as resp:
            assert resp.status_code == 200
            ct = resp.headers.get("content-type", "").lower()
            assert ct.startswith("text/event-stream")
            lines = list(resp.iter_lines())
            assert any(l.startswith("data: ") and "[DONE]" not in l for l in lines)
            assert sum(1 for l in lines if l.strip().lower() == "data: [done]") == 1
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        import tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter as anthropic_mod
        monkeypatch.setattr(anthropic_mod, "http_client_factory", lambda *a, **k: _FakeClient())

        client = client_user_only
        with client.stream("POST", "/api/v1/chat/completions", json=_payload(stream=True), headers=_AUTH_HEADERS) as resp:
            assert resp.status_code == 200
            ct = resp.headers.get("content-type", "").lower()
            assert ct.startswith("text/event-stream")
            lines = list(resp.iter_lines())
            assert any(l.startswith("data: ") and "[DONE]" not in l for l in lines)
            assert sum(1 for l in lines if l.strip().lower() == "data: [done]") == 1
