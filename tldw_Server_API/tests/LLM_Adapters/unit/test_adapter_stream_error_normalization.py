from __future__ import annotations

"""
Unit tests to verify adapter-native HTTP stream paths normalize httpx errors
into the project Chat*Error exceptions. Complements endpoint SSE error tests.
"""

from typing import Any, Dict, Type
import pytest

pytestmark = pytest.mark.usefixtures("no_live_network", "loopback_custom_openai_endpoints")


def _httpx_status_error(status_code: int):
    import httpx
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp = httpx.Response(status_code, request=req, content=b'{"error":{"message":"x"}}')
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return e
    raise AssertionError("Expected HTTPStatusError not raised")


class _FakeResponse:
    def __init__(self, status_code: int):
        self._err = _httpx_status_error(status_code)

    def raise_for_status(self):
        raise self._err

    def iter_lines(self):  # pragma: no cover - not reached on error
        yield "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n"
        yield "data: [DONE]\n\n"


class _FakeStreamCtx:
    def __init__(self, r: _FakeResponse):
        self._r = r

    def __enter__(self):
        return self._r

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - passthrough
        return False


class _FakeClient:
    def __init__(self, status_code: int = 400, *args, **kwargs):
        self._status = status_code

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover
        return False

    def post(self, *args, **kwargs):  # pragma: no cover - chat() path not used here
        return _FakeResponse(self._status)

    def stream(self, *args, **kwargs):
        return _FakeStreamCtx(_FakeResponse(self._status))


@pytest.mark.parametrize(
    "provider_key, adapter_cls_path, status_code, expected_err",
    [
        # Core providers across representative statuses
        ("openai", "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.OpenAIAdapter", 400, "ChatBadRequestError"),
        ("openai", "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.OpenAIAdapter", 404, "ChatBadRequestError"),
        ("openai", "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.OpenAIAdapter", 422, "ChatBadRequestError"),
        ("openai", "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.OpenAIAdapter", 401, "ChatAuthenticationError"),
        ("openai", "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.OpenAIAdapter", 403, "ChatAuthenticationError"),
        ("openai", "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.OpenAIAdapter", 429, "ChatRateLimitError"),
        ("openai", "tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter.OpenAIAdapter", 500, "ChatProviderError"),

        ("anthropic", "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.AnthropicAdapter", 400, "ChatBadRequestError"),
        ("anthropic", "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.AnthropicAdapter", 403, "ChatAuthenticationError"),
        ("anthropic", "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.AnthropicAdapter", 429, "ChatRateLimitError"),
        ("anthropic", "tldw_Server_API.app.core.LLM_Calls.providers.anthropic_adapter.AnthropicAdapter", 502, "ChatProviderError"),

        ("groq", "tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter.GroqAdapter", 400, "ChatBadRequestError"),
        ("groq", "tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter.GroqAdapter", 401, "ChatAuthenticationError"),
        ("groq", "tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter.GroqAdapter", 429, "ChatRateLimitError"),
        ("groq", "tldw_Server_API.app.core.LLM_Calls.providers.groq_adapter.GroqAdapter", 503, "ChatProviderError"),

        ("openrouter", "tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter.OpenRouterAdapter", 404, "ChatBadRequestError"),
        ("openrouter", "tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter.OpenRouterAdapter", 401, "ChatAuthenticationError"),
        ("openrouter", "tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter.OpenRouterAdapter", 429, "ChatRateLimitError"),
        ("openrouter", "tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter.OpenRouterAdapter", 500, "ChatProviderError"),

        ("google", "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.GoogleAdapter", 400, "ChatBadRequestError"),
        ("google", "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.GoogleAdapter", 401, "ChatAuthenticationError"),
        ("google", "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.GoogleAdapter", 429, "ChatRateLimitError"),
        ("google", "tldw_Server_API.app.core.LLM_Calls.providers.google_adapter.GoogleAdapter", 500, "ChatProviderError"),

        ("mistral", "tldw_Server_API.app.core.LLM_Calls.providers.mistral_adapter.MistralAdapter", 403, "ChatAuthenticationError"),
        ("mistral", "tldw_Server_API.app.core.LLM_Calls.providers.mistral_adapter.MistralAdapter", 422, "ChatBadRequestError"),

        # Stage 3 providers
        ("qwen", "tldw_Server_API.app.core.LLM_Calls.providers.qwen_adapter.QwenAdapter", 400, "ChatBadRequestError"),
        ("qwen", "tldw_Server_API.app.core.LLM_Calls.providers.qwen_adapter.QwenAdapter", 401, "ChatAuthenticationError"),
        ("qwen", "tldw_Server_API.app.core.LLM_Calls.providers.qwen_adapter.QwenAdapter", 429, "ChatRateLimitError"),
        ("qwen", "tldw_Server_API.app.core.LLM_Calls.providers.qwen_adapter.QwenAdapter", 500, "ChatProviderError"),

        ("deepseek", "tldw_Server_API.app.core.LLM_Calls.providers.deepseek_adapter.DeepSeekAdapter", 400, "ChatBadRequestError"),
        ("deepseek", "tldw_Server_API.app.core.LLM_Calls.providers.deepseek_adapter.DeepSeekAdapter", 403, "ChatAuthenticationError"),
        ("deepseek", "tldw_Server_API.app.core.LLM_Calls.providers.deepseek_adapter.DeepSeekAdapter", 429, "ChatRateLimitError"),
        ("deepseek", "tldw_Server_API.app.core.LLM_Calls.providers.deepseek_adapter.DeepSeekAdapter", 500, "ChatProviderError"),

        ("huggingface", "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter.HuggingFaceAdapter", 401, "ChatAuthenticationError"),
        ("huggingface", "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter.HuggingFaceAdapter", 422, "ChatBadRequestError"),
        ("huggingface", "tldw_Server_API.app.core.LLM_Calls.providers.huggingface_adapter.HuggingFaceAdapter", 500, "ChatProviderError"),

        ("custom_openai", "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.CustomOpenAIAdapter", 400, "ChatBadRequestError"),
        ("custom_openai", "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.CustomOpenAIAdapter", 401, "ChatAuthenticationError"),
        ("custom_openai", "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.CustomOpenAIAdapter", 429, "ChatRateLimitError"),
        ("custom_openai", "tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter.CustomOpenAIAdapter", 500, "ChatProviderError"),
    ],
)
def test_adapter_stream_normalizes_httpx_errors(monkeypatch, provider_key: str, adapter_cls_path: str, status_code: int, expected_err: str):
    # Force native HTTP path (under pytest adapters typically opt-in already)
    monkeypatch.setenv("STREAMS_UNIFIED", "1")
    try:
        monkeypatch.setenv(f"LLM_ADAPTERS_NATIVE_HTTP_{provider_key.upper()}", "1")
    except Exception:
        # Some providers use other flags; adapters prefer native path when adapters are enabled in tests
        _ = None

    # Import adapter class dynamically
    parts = adapter_cls_path.split(".")
    mod_path = ".".join(parts[:-1])
    cls_name = parts[-1]
    mod = __import__(mod_path, fromlist=[cls_name])
    Adapter: Type[Any] = getattr(mod, cls_name)

    # Patch client factory in module where the adapter is defined
    # Prefer a named factory attribute; otherwise, fallback to module-level _hc_create_client
    if hasattr(mod, "http_client_factory"):
        monkeypatch.setattr(mod, "http_client_factory", lambda *a, **k: _FakeClient(status_code), raising=True)
    elif hasattr(mod, "_hc_create_client"):
        monkeypatch.setattr(mod, "_hc_create_client", lambda *a, **k: _FakeClient(status_code), raising=True)
    else:
        # Last resort: patch the shared http client create function
        import tldw_Server_API.app.core.http_client as http_client_mod
        monkeypatch.setattr(http_client_mod, "create_client", lambda *a, **k: _FakeClient(status_code), raising=True)

    adapter = Adapter()
    req: Dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}], "model": "x", "api_key": "k", "stream": True}
    with pytest.raises(Exception) as ei:
        # Trigger the generator body to execute raise_for_status()
        list(adapter.stream(req))
    assert ei.value.__class__.__name__ == expected_err
