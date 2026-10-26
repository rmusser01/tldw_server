from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_obj: Dict[str, Any] | None = None, lines: List[str] | None = None):
        self.status_code = status_code
        self._json = json_obj or {}
        self._lines = lines or []

    def raise_for_status(self):
        if 400 <= self.status_code:
            import httpx
            request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self):
        return self._json

    def iter_lines(self):
        for l in self._lines:
            yield l


class _FakeStreamCtx:
    def __init__(self, response: _FakeResponse):
        self._resp = response

    def __enter__(self):
        return self._resp

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.last_post = None
        self.last_stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url: str, json: Dict[str, Any], headers: Dict[str, str]):
        self.last_post = {"url": url, "json": json, "headers": headers}
        return _FakeResponse(200, {"object": "chat.completion", "choices": [{"message": {"content": "ok"}}]})

    def stream(self, method: str, url: str, json: Dict[str, Any], headers: Dict[str, str]):
        self.last_stream = {"method": method, "url": url, "json": json, "headers": headers}
        lines = [
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}",
            "data: [DONE]",
        ]
        return _FakeStreamCtx(_FakeResponse(200, lines=lines))


@pytest.fixture(autouse=True)
def _enable_native_http(monkeypatch):
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_OPENAI", "1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    # These take precedence over OPENAI_BASE_URL (OpenAIAdapter._openai_base_url).
    monkeypatch.delenv("OPENAI_API_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    yield


def test_openai_adapter_native_http_non_streaming(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
    # Patch adapter factory to return our fake client
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod
    monkeypatch.setattr(openai_mod, "http_client_factory", lambda *a, **k: _FakeClient(*a, **k))

    adapter = OpenAIAdapter()
    req = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "gpt-4o-mini",
        "api_key": "sk-test",
        "temperature": 0.1,
    }
    resp = adapter.chat(req)
    assert resp.get("object") == "chat.completion"


def test_openai_adapter_native_http_streaming(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod
    monkeypatch.setattr(openai_mod, "http_client_factory", lambda *a, **k: _FakeClient(*a, **k))

    adapter = OpenAIAdapter()
    req = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-4o-mini",
        "api_key": "sk-test",
        "temperature": 0.2,
        "stream": True,
    }
    chunks = list(adapter.stream(req))
    # Should produce SSE lines with double newlines
    assert any(c.startswith("data: ") for c in chunks)
    assert sum(1 for c in chunks if "[DONE]" in c) == 1


def test_openai_adapter_native_http_logprobs_forwarded(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod

    captured = {}

    def _factory(*args, **kwargs):
        client = _FakeClient(*args, **kwargs)
        captured["client"] = client
        return client

    monkeypatch.setattr(openai_mod, "http_client_factory", _factory, raising=True)

    adapter = OpenAIAdapter()
    req = {
        "messages": [{"role": "user", "content": "logprobs please"}],
        "model": "gpt-4o-mini",
        "api_key": "sk-test",
        "logprobs": True,
        "top_logprobs": 4,
    }
    _ = adapter.chat(req)
    payload = captured["client"].last_post["json"]
    assert payload.get("logprobs") is True
    assert payload.get("top_logprobs") == 4


def test_openai_adapter_merges_extra_body_and_headers(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod

    captured: Dict[str, Any] = {}

    def _factory(*args, **kwargs):
        client = _FakeClient(*args, **kwargs)
        captured["client"] = client
        return client

    monkeypatch.setattr(openai_mod, "http_client_factory", _factory, raising=True)

    adapter = OpenAIAdapter()
    req = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-4o-mini",
        "api_key": "sk-test",
        "temperature": 0.1,
        "extra_body": {"temperature": 0.9, "x_extra": "y"},
        "extra_headers": {
            "X-Test": "1",
            "Authorization": "Bearer override",
            "content-type": "text/plain",
        },
    }
    _ = adapter.chat(req)
    payload = captured["client"].last_post["json"]
    headers = captured["client"].last_post["headers"]
    assert payload.get("temperature") == 0.1
    assert payload.get("x_extra") == "y"
    assert headers.get("Authorization") == "Bearer sk-test"
    assert headers.get("Content-Type") == "application/json"
    assert headers.get("X-Test") == "1"
    assert "content-type" not in headers


def _streamed_payload(monkeypatch, **request_fields: Any) -> dict[str, Any]:
    """Stream one request through a fake client and return the JSON body the adapter sent."""
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod

    clients: list[_FakeClient] = []

    def _factory(*args, **kwargs):
        client = _FakeClient(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(openai_mod, "http_client_factory", _factory, raising=True)
    request = {"messages": [{"role": "user", "content": "hello"}], "model": "gpt-4o-mini", "api_key": "sk-test"}
    list(openai_mod.OpenAIAdapter().stream({**request, **request_fields}))
    return clients[0].last_stream["json"]


@pytest.mark.parametrize(
    "base_url",
    ["https://api.openai.com/v1", "https://eu.api.openai.com/v1"],
    ids=["global", "data-residency-region"],
)
def test_an_openai_stream_asks_openai_for_its_usage_chunk(monkeypatch, base_url):
    """OpenAI sends a stream's usage only when asked, so a stream to OpenAI's own hosts asks (TASK-13951)."""
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    payload = _streamed_payload(monkeypatch)

    assert payload["stream_options"] == {"include_usage": True}


def test_the_callers_stream_options_win_over_the_usage_request(monkeypatch):
    payload = _streamed_payload(
        monkeypatch, extra_body={"stream_options": {"include_usage": False, "include_obfuscation": False}}
    )

    assert payload["stream_options"] == {"include_usage": False, "include_obfuscation": False}


@pytest.mark.parametrize(
    "base_url",
    ["https://llm.example.internal/v1", "https://api.openai.com.example.net/v1"],
    ids=["other-server", "lookalike-host"],
)
def test_an_openai_provider_stream_to_another_server_is_not_asked_for_usage(monkeypatch, base_url):
    """Another server behind the openai provider may reject stream_options (TASK-13951)."""
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    payload = _streamed_payload(monkeypatch)

    assert "stream_options" not in payload
