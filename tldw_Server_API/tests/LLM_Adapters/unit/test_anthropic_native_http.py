from __future__ import annotations

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
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            resp = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError("err", request=req, response=resp)

    def json(self):
        return self._json

    def iter_lines(self):
        for l in self._lines:
            yield l


class _FakeStreamCtx:
    def __init__(self, response: _FakeResponse):
        self._r = response

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

    def post(self, url: str, json: Dict[str, Any], headers: Dict[str, str]):
        return _FakeResponse(200, {"object": "chat.completion", "choices": [{"message": {"content": "ok"}}]})

    def stream(self, method: str, url: str, json: Dict[str, Any], headers: Dict[str, str]):
        lines = [
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"a\"}}",
            "data: [DONE]",
        ]
        return _FakeStreamCtx(_FakeResponse(200, lines=lines))


@pytest.fixture(autouse=True)
def _enable_native(monkeypatch):
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_ANTHROPIC", "1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
    yield


def test_anthropic_adapter_native_http_non_streaming(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter import AnthropicAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter as anth_mod
    monkeypatch.setattr(anth_mod, "http_client_factory", lambda *a, **k: _FakeClient(*a, **k))
    a = AnthropicAdapter()
    r = a.chat({"messages": [{"role": "user", "content": "hi"}], "model": "claude-sonnet", "api_key": "k", "max_tokens": 32})
    assert r.get("object") == "chat.completion"


def test_anthropic_adapter_native_http_streaming(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter import AnthropicAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter as anth_mod
    monkeypatch.setattr(anth_mod, "http_client_factory", lambda *a, **k: _FakeClient(*a, **k))
    a = AnthropicAdapter()
    chunks = list(a.stream({"messages": [{"role": "user", "content": "hi"}], "model": "claude-sonnet", "api_key": "k", "max_tokens": 32, "stream": True}))
    assert any(c.startswith("data: ") for c in chunks)
    assert sum(1 for c in chunks if "[DONE]" in c) == 1


def test_anthropic_adapter_usage_counts_and_keeps_prompt_cache_tokens(monkeypatch):
    """input_tokens is only the uncached remainder: prompt_tokens adds the cache fields, which stay in the block (TASK-13858)."""
    import tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter as anth_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter import AnthropicAdapter
    from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage

    message = {
        "type": "message",
        "id": "msg-cached",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 60000, "output_tokens": 100},
    }

    class _CachedClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            return _FakeResponse(200, message)

    monkeypatch.setattr(anth_mod, "http_client_factory", lambda *a, **k: _CachedClient())
    r = AnthropicAdapter().chat({"messages": [{"role": "user", "content": "hi"}], "model": "claude-sonnet-4-6", "api_key": "k", "max_tokens": 32})

    assert r["usage"] == {
        "prompt_tokens": 62003,
        "completion_tokens": 100,
        "total_tokens": 62103,
        "cache_creation_input_tokens": 2000,
        "cache_read_input_tokens": 60000,
    }
    # What the chat pipeline and the provider recorder cost: only the 3 uncached tokens at the input rate.
    normalized = normalize_llm_usage(provider="anthropic", usage=r["usage"])
    assert (normalized.billable_input_tokens, normalized.cache_read_input_tokens, normalized.cache_write_input_tokens) == (3, 60000, 2000)


def test_anthropic_adapter_keeps_the_cache_creation_breakdown(monkeypatch):
    """The 5-minute and 1-hour split of cache writes stays in the usage block, so 1-hour writes are costed at 2x (TASK-13878)."""
    import tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter as anth_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter import AnthropicAdapter
    from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage

    split = {"ephemeral_5m_input_tokens": 500, "ephemeral_1h_input_tokens": 1500}
    message = {
        "type": "message",
        "id": "msg-1h",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "cache_creation_input_tokens": 2000, "cache_creation": split, "output_tokens": 100},
    }

    class _Client(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            return _FakeResponse(200, message)

    monkeypatch.setattr(anth_mod, "http_client_factory", lambda *a, **k: _Client())
    r = AnthropicAdapter().chat({"messages": [{"role": "user", "content": "hi"}], "model": "claude-sonnet-4-6", "api_key": "k", "max_tokens": 32})

    assert r["usage"]["cache_creation"] == split
    assert normalize_llm_usage(provider="anthropic", usage=r["usage"]).cache_write_1h_input_tokens == 1500


def test_anthropic_adapter_keeps_speed_and_inference_geo(monkeypatch):
    """Fast mode and US-only inference stay in the usage block, so the chat path costs them too (TASK-13885)."""
    import tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter as anth_mod
    from tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter import AnthropicAdapter

    message = {
        "type": "message", "id": "msg-fast", "model": "claude-opus-5", "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 10, "speed": "fast", "inference_geo": "us"},
    }

    class _Client(_FakeClient):
        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            return _FakeResponse(200, message)

    monkeypatch.setattr(anth_mod, "http_client_factory", lambda *a, **k: _Client())
    r = AnthropicAdapter().chat({"messages": [{"role": "user", "content": "hi"}], "model": "claude-opus-5", "api_key": "k", "max_tokens": 32})

    assert (r["usage"]["speed"], r["usage"]["inference_geo"]) == ("fast", "us")
