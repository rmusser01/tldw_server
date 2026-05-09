from __future__ import annotations

from typing import Any, Dict, List

import pytest


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_obj: Dict[str, Any] | None = None, lines: List[str] | None = None):
        self.status_code = status_code
        self._json = json_obj or {"object": "chat.completion", "choices": [{"message": {"content": "ok"}}]}
        self._lines = lines or [
            "data: chunk",
            "data: [DONE]",
        ]

    def raise_for_status(self):
        if 400 <= self.status_code:
            import httpx
            req = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
            resp = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError("err", request=req, response=resp)

    def json(self):
        return self._json

    def iter_lines(self):
        for l in self._lines:
            yield l


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

    def post(self, url: str, json: Dict[str, Any], headers: Dict[str, str]):
        assert "chat/completions" in url
        return _FakeResponse(200)

    def stream(self, method: str, url: str, json: Dict[str, Any], headers: Dict[str, str]):
        return _FakeStreamCtx(_FakeResponse(200))


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_CUSTOM_OPENAI", "1")
    monkeypatch.setenv("LOGURU_LEVEL", "ERROR")
    yield


def test_custom_openai_adapter_native_http_non_streaming(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import CustomOpenAIAdapter
    import tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter as co_mod
    monkeypatch.setattr(co_mod, "http_client_factory", lambda *a, **k: _FakeClient(*a, **k))
    a = CustomOpenAIAdapter()
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "my-model",
        "api_key": "k",
        "app_config": {"custom_openai_api": {"api_ip": "http://127.0.0.1:11434/v1"}},
    }
    r = a.chat(request)
    assert r.get("object") == "chat.completion"


def test_custom_openai_adapter_native_http_streaming(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import CustomOpenAIAdapter2
    import tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter as co_mod
    monkeypatch.setattr(co_mod, "http_client_factory", lambda *a, **k: _FakeClient(*a, **k))
    a = CustomOpenAIAdapter2()
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "my-model",
        "api_key": "k",
        "app_config": {"custom_openai_api_2": {"api_ip": "http://127.0.0.1:11434"}},
        "stream": True,
    }
    chunks = list(a.stream(request))
    assert any(c.startswith("data: ") for c in chunks)
    assert sum(1 for c in chunks if "[DONE]" in c) == 1


def test_custom_openai_adapter_fallback_accepts_canonical_env(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import CustomOpenAIAdapter

    monkeypatch.setenv("CUSTOM_OPENAI_API_IP", "http://127.0.0.1:8000/v1")
    adapter = CustomOpenAIAdapter()

    assert adapter._resolve_base({"app_config": {}}) == "http://127.0.0.1:8000/v1"


def test_custom_openai_adapter_fallback_preserves_numbered_env_compatibility(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import CustomOpenAIAdapter2

    monkeypatch.setenv("CUSTOM_OPENAI_API_IP_2", "http://127.0.0.1:8002/v1")
    adapter = CustomOpenAIAdapter2()

    assert adapter._resolve_base({"app_config": {}}) == "http://127.0.0.1:8002/v1"


def test_custom_openai_adapter_factory_supports_numbered_endpoint_env(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry

    monkeypatch.setenv("CUSTOM_OPENAI_API_IP_37", "http://127.0.0.1:8037/v1")

    registry = ChatProviderRegistry()
    adapter = registry.get_adapter("custom-openai-api-37")

    assert adapter is not None
    assert adapter.name == "custom-openai-api-37"
    assert adapter.config_section == "custom_openai_api_37"
    assert adapter._resolve_base({"app_config": {}}) == "http://127.0.0.1:8037/v1"


def test_numbered_custom_openai_adapters_require_explicit_base_url(monkeypatch):
    from tldw_Server_API.app.core.custom_openai_providers import custom_openai_endpoint_env_keys
    from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import (
        CustomOpenAIAdapter2,
        make_custom_openai_adapter_class,
    )

    for env_key in (*custom_openai_endpoint_env_keys(2), *custom_openai_endpoint_env_keys(37)):
        monkeypatch.delenv(env_key, raising=False)

    with pytest.raises(RuntimeError, match="requires an explicit base URL"):
        CustomOpenAIAdapter2()._resolve_base({"app_config": {}})

    with pytest.raises(RuntimeError, match="requires an explicit base URL"):
        make_custom_openai_adapter_class(37)()._resolve_base({"app_config": {}})
