from __future__ import annotations

"""
Adapter-level mid-stream failure behavior: ensure adapters raise a Chat*Error
when the HTTP streaming iteration fails after some chunks. Endpoint-level
tests cover transforming such exceptions into SSE error frames.
"""

from typing import Any, Dict, List, Type
import pytest


class _FakeResponse:
    def __init__(self):
        self._lines = [
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
        ]

    def raise_for_status(self):
        return None

    def iter_lines(self):
        import httpx
        for ln in self._lines:
            yield ln
        # Simulate mid-stream transport error
        raise httpx.ReadError("mid-stream failure")


class _FakeStreamCtx:
    def __init__(self):
        self._r = _FakeResponse()

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

    def stream(self, *args, **kwargs):
        return _FakeStreamCtx()

    def post(self, *a, **k):  # pragma: no cover - not used here
        raise RuntimeError("not used")


@pytest.mark.parametrize(
    "adapter_path",
    [
        "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.OpenAIAdapter",
        "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.AnthropicAdapter",
        "tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter.GroqAdapter",
        "tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter.OpenRouterAdapter",
        "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.GoogleAdapter",
        "tldw_Server_API.app.core.LLM_Calls.providers.mistral_adapter.MistralAdapter",
        "tldw_Server_API.app.core.LLM_Calls.providers.qwen_adapter.QwenAdapter",
        "tldw_Server_API.app.core.LLM_Calls.providers.deepseek_adapter.DeepSeekAdapter",
        "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter.HuggingFaceAdapter",
        "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.CustomOpenAIAdapter",
        "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.CustomOpenAIAdapter2",
    ],
)
def test_adapter_midstream_error_raises(monkeypatch, adapter_path: str):
    parts = adapter_path.split(".")
    mod_path = ".".join(parts[:-1])
    cls_name = parts[-1]
    mod = __import__(mod_path, fromlist=[cls_name])
    Adapter: Type[Any] = getattr(mod, cls_name)

    # Patch httpx factory alias if present, else module-level create function
    if hasattr(mod, "http_client_factory"):
        monkeypatch.setattr(mod, "http_client_factory", lambda *a, **k: _FakeClient(), raising=True)
    elif hasattr(mod, "_hc_create_client"):
        monkeypatch.setattr(mod, "_hc_create_client", lambda *a, **k: _FakeClient(), raising=True)
    else:
        import tldw_Server_API.app.core.http_client as hc
        monkeypatch.setattr(hc, "create_client", lambda *a, **k: _FakeClient(), raising=True)

    adapter = Adapter()
    req: Dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}], "model": "x", "api_key": "k", "stream": True}
    if cls_name == "CustomOpenAIAdapter2":
        req["base_url"] = "http://mock-custom-openai.local/v1"

    with pytest.raises(Exception) as ei:
        list(adapter.stream(req))
    # The adapter should map non-HTTP transport errors to ChatProviderError
    assert ei.value.__class__.__name__ in {"ChatProviderError", "ChatAPIError"}
